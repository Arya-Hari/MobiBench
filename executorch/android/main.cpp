/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

 #include <gflags/gflags.h>
 #include <chrono>
 #include <fstream>
 #include <iomanip>
 #include <iostream>
 #include <sstream>
 #include <string>
 #include <vector>
 #include <algorithm> // Required for std::remove
 #include <optional>
 #include <regex>

 #include <executorch/examples/models/llama/runner/runner.h>
 #include "csv.h" // The CSV parsing library we added

 #if defined(ET_USE_THREADPOOL)
 #include <executorch/extension/threadpool/cpuinfo_utils.h>
 #include <executorch/extension/threadpool/threadpool.h>
 #endif

 // --- Command-line flags for evaluation ---

 DEFINE_string(model_path, "llama2.pte", "Model serialized in flatbuffer format.");
 DEFINE_string(tokenizer_path, "tokenizer.bin", "The path to the tokenizer model.");
 DEFINE_string(data_paths, "", "Optional: Comma-separated data files for the model.");

 // New flags for dataset evaluation
 DEFINE_string(dataset_type, "", "Type of dataset: context_qa, summarization, scientific_mcq, or mmlu_mcq.");
 DEFINE_string(csv_path, "", "The path to the input dataset CSV file.");
 DEFINE_string(output_json, "results.jsonl", "The path to write the output JSONL file.");

 // Standard generation flags
 DEFINE_double(temperature, 0.8f, "Temperature for sampling. 0 = greedy. Default is 0.8f.");
 DEFINE_int32(seq_len, 128, "Maximum number of tokens to generate.");
 DEFINE_int32(cpu_threads, -1, "Number of CPU threads. -1 for auto-detection.");

 // --- Helper Functions ---

 // Helper to escape strings for proper JSON output
 std::string escape_json(const std::string &s) {
     std::ostringstream o;
     for (char c : s) {
         switch (c) {
             case '"': o << "\\\""; break;
             case '\\': o << "\\\\"; break;
             case '\b': o << "\\b"; break;
             case '\f': o << "\\f"; break;
             case '\n': o << "\\n"; break;
             case '\r': o << "\\r"; break;
             case '\t': o << "\\t"; break;
             default:
                 if ('\x00' <= c && c <= '\x1f') {
                     o << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(static_cast<unsigned char>(c));
                 } else {
                     o << c;
                 }
         }
     }
     return o.str();
 }

 // Parses comma-separated string lists
 std::vector<std::string> parseStringList(const std::string& input) {
   std::vector<std::string> result;
   if (input.empty()) return result;
   std::stringstream ss(input);
   std::string item;
   while (std::getline(ss, item, ',')) {
     item.erase(0, item.find_first_not_of(" \t"));
     item.erase(item.find_last_not_of(" \t") + 1);
     if (!item.empty()) result.push_back(item);
   }
   return result;
 }

 // -----------------------------------------------------------------------------
 // Robust multiline-safe CSV helper used by context_qa and MMLU
 // -----------------------------------------------------------------------------

 std::vector<std::string> parse_csv_line(const std::string& line) {
     std::vector<std::string> result;
     std::string current;
     bool in_quotes = false;

     for (size_t i = 0; i < line.size(); ++i) {
         char c = line[i];

         if (c == '"') {
             if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                 current += '"';  // escaped quote
                 ++i;
             } else {
                 in_quotes = !in_quotes;
             }
         } else if (c == ',' && !in_quotes) {
             result.push_back(current);
             current.clear();
         } else {
             current += c;
         }
     }
     result.push_back(current);
     return result;
 }

 // Reads CSV file and returns header cols and the parsed rows (robust to newlines inside quotes).
 bool read_csv_rows_multiline(
     const std::string& csv_path,
     std::vector<std::string>& out_header,
     std::vector<std::vector<std::string>>& out_rows) {

     std::ifstream file(csv_path);
     if (!file.is_open()) {
         std::cerr << "❌ Failed to open CSV file: " << csv_path << std::endl;
         return false;
     }

     std::string header_line;
     if (!std::getline(file, header_line)) {
         std::cerr << "❌ Empty CSV or failed to read header: " << csv_path << std::endl;
         return false;
     }

     out_header = parse_csv_line(header_line);
     for (auto &h : out_header) {
         // normalize header by trimming quotes/spaces
         if (!h.empty() && h.front() == '"' && h.back() == '"') {
             h = h.substr(1, h.size() - 2);
         }
         // lower-case and trim
         size_t s = h.find_first_not_of(" \t");
         size_t e = h.find_last_not_of(" \t");
         if (s != std::string::npos && e != std::string::npos) {
             h = h.substr(s, e - s + 1);
         }
     }

     std::string line;
     std::string buffer;

     while (std::getline(file, line)) {
         buffer += line;
         int quote_count = std::count(buffer.begin(), buffer.end(), '"');

         // If quotes are unbalanced, we assume a newline inside a quoted field; keep reading
         if (quote_count % 2 != 0) {
             buffer += '\n';
             continue;
         }

         auto cols = parse_csv_line(buffer);
         buffer.clear();
         out_rows.push_back(std::move(cols));
     }

     return true;
 }

 // -----------------------------------------------------------------------------
 // Context QA helper
 // -----------------------------------------------------------------------------
 struct QARecord {
     std::string document_id;
     std::string document_topic;
     std::string document_path;
     std::string document_extracted;
     std::string question_id;
     std::string question;
     std::string answer;
     std::string long_answer;
 };

 std::vector<QARecord> read_context_qa_csv(const std::string& csv_path) {
     std::ifstream file(csv_path);
     if (!file.is_open()) {
         std::cerr << "❌ Failed to open CSV file: " << csv_path << std::endl;
         return {};
     }

     std::vector<QARecord> records;
     std::string line;
     std::string buffer;

     // Skip header
     std::getline(file, line);

     while (std::getline(file, line)) {
         buffer += line;
         int quote_count = std::count(buffer.begin(), buffer.end(), '"');

         if (quote_count % 2 != 0) {
             buffer += '\n';
             continue;
         }

         auto cols = parse_csv_line(buffer);
         buffer.clear();

         if (cols.size() < 8) {
             std::cerr << "⚠️ Skipping malformed row with " << cols.size() << " columns." << std::endl;
             continue;
         }

         QARecord rec {
             cols[0], cols[1], cols[2], cols[3],
             cols[4], cols[5], cols[6], cols[7]
         };
         records.push_back(rec);
     }

     std::cout << "✅ Loaded " << records.size() << " context_qa rows successfully.\n";
     return records;
 }

 // --- Main Evaluation Logic ---

 void run_evaluation(
     ::executorch::extension::llm::TextLLMRunner* runner,
     const std::string& dataset_type,
     const std::string& csv_path,
     std::ofstream& output_file) {

     int row_index = 0;
     try {
         if (dataset_type == "context_qa") {
             auto qa_records = read_context_qa_csv(csv_path);
             for (const auto& rec : qa_records) {
                 std::string prompt = "Answer the question with the given context.\n\nContext:\n" + rec.document_extracted +
                                      "\n\nQuestion:\n" + rec.question + "\n\nProvide in short answer.\n\nAnswer:";

                 std::cout << "Processing entry " << ++row_index << "..." << std::endl;

                 // Collect model output
                 std::stringstream model_output_stream;

                 // Use std::optional for stats (avoid copy-assignment)
                 std::optional<::executorch::extension::llm::Stats> stats_snapshot;
                 // token callback appends to model_output_stream
                 auto token_callback = [&](const std::string& piece) {
                     model_output_stream << piece;
                 };

                 // stats callback captures stats into local snapshot using emplace
                 auto stats_callback = [&](const ::executorch::extension::llm::Stats& s) {
                     stats_snapshot.emplace(s);
                 };

                 runner->reset();
                 ::executorch::extension::llm::GenerationConfig config {
                     .max_new_tokens = static_cast<int32_t>(FLAGS_seq_len),
                     .temperature = static_cast<float>(FLAGS_temperature),
                 };

                 auto start_time = std::chrono::high_resolution_clock::now();
                 auto error = runner->generate(prompt, config, token_callback, stats_callback);
                 auto end_time = std::chrono::high_resolution_clock::now();

                 if (error != executorch::runtime::Error::Ok) {
                     ET_LOG(Error, "Inference failed for entry %d", row_index);
                     // Still write a JSON entry marking failure
                     double wallclock_s = std::chrono::duration<double>(end_time - start_time).count();
                     output_file << "{\"id\": " << row_index
                                 << ", \"prompt\": \"" << escape_json(prompt)
                                 << "\", \"model_output\": \"\", \"logs\": \"\", \"timings\": {}, \"system_metrics\": {}, \"wallclock_s\": " << wallclock_s << "}\n";
                     continue;
                 }

                 std::chrono::duration<double> wallclock_s = end_time - start_time;
                 std::string model_output = model_output_stream.str();

                 // Build timings JSON
                 double model_load_s = -1.0;
                 double inference_s = -1.0;
                 double first_token_s = -1.0;
                 int num_prompt_tokens = -1;
                 int num_generated_tokens = -1;

                 if (stats_snapshot.has_value()) {
                     auto &s = stats_snapshot.value();
                     model_load_s = (s.model_load_end_ms - s.model_load_start_ms) / 1000.0;
                     inference_s = (s.inference_end_ms - s.inference_start_ms) / 1000.0;
                     first_token_s = (s.first_token_ms - s.inference_start_ms) / 1000.0;
                     num_prompt_tokens = static_cast<int>(s.num_prompt_tokens);
                     num_generated_tokens = static_cast<int>(s.num_generated_tokens);
                 }

                 std::string logs_str = "";
                 output_file << "{";
                 output_file << "\"id\": " << row_index << ", ";
                 output_file << "\"prompt\": \"" << escape_json(prompt) << "\", ";
                 output_file << "\"model_output\": \"" << escape_json(model_output) << "\", ";
                 output_file << "\"logs\": \"" << escape_json(logs_str) << "\", ";
                 output_file << "\"timings\": {";
                 output_file << "\"model_load_s\": " << std::fixed << std::setprecision(6) << model_load_s << ", ";
                 output_file << "\"inference_s\": " << std::fixed << std::setprecision(6) << inference_s << ", ";
                 output_file << "\"first_token_s\": " << std::fixed << std::setprecision(6) << first_token_s << ", ";
                 output_file << "\"num_prompt_tokens\": " << num_prompt_tokens << ", ";
                 output_file << "\"num_generated_tokens\": " << num_generated_tokens;
                 output_file << "}, ";
                 output_file << "\"system_metrics\": {";
                 output_file << "\"rss_mb\": null";
                 output_file << "}, ";
                 output_file << "\"wallclock_s\": " << std::fixed << std::setprecision(6) << wallclock_s.count();
                 output_file << "}\n";
             }

         } else if (dataset_type == "summarization") {
            io::CSVReader<3> in(csv_path);
            in.read_header(io::ignore_extra_column, "article", "highlights", "id");
            std::string article, highlights_unused, id_unused;
        
            while (in.read_row(article, highlights_unused, id_unused)) {
                std::string prompt = "Given the below text, summarize it.\n\nText:\n" + article + "\n\nSummary:";
        
                std::cout << "Processing entry " << ++row_index << "..." << std::endl;
                runner->reset();
        
                ::executorch::extension::llm::GenerationConfig config{
                    .max_new_tokens = static_cast<int32_t>(FLAGS_seq_len),
                    .temperature = static_cast<float>(FLAGS_temperature),
                };
        
                std::stringstream model_output_stream;
                auto token_callback = [&](const std::string &piece) {
                    model_output_stream << piece;
                };
        
                std::optional<::executorch::extension::llm::Stats> stats_snapshot;
                auto stats_callback = [&](const ::executorch::extension::llm::Stats &s) {
                    stats_snapshot.emplace(s);
                };
        
                auto start_time = std::chrono::high_resolution_clock::now();
                auto err = runner->generate(prompt, config, token_callback, stats_callback);
                auto end_time = std::chrono::high_resolution_clock::now();
        
                std::chrono::duration<double> wallclock_s = end_time - start_time;
        
                if (err != executorch::runtime::Error::Ok) {
                    ET_LOG(Error, "Inference failed for entry %d", row_index);
                    output_file << "{"
                                << "\"id\": " << row_index << ", "
                                << "\"prompt\": \"" << escape_json(prompt) << "\", "
                                << "\"model_output\": \"\", "
                                << "\"logs\": \"\", "
                                << "\"timings\": {}, "
                                << "\"system_metrics\": {}, "
                                << "\"wallclock_s\": " << std::fixed << std::setprecision(6)
                                << wallclock_s.count()
                                << "}\n";
                    continue;
                }
        
                std::string model_output = model_output_stream.str();
        
                double model_load_s = -1.0;
                double inference_s = -1.0;
                double first_token_s = -1.0;
                int num_prompt_tokens = -1;
                int num_generated_tokens = -1;
        
                if (stats_snapshot.has_value()) {
                    auto &s = stats_snapshot.value();
                    model_load_s = (s.model_load_end_ms - s.model_load_start_ms) / 1000.0;
                    inference_s = (s.inference_end_ms - s.inference_start_ms) / 1000.0;
                    first_token_s = (s.first_token_ms - s.inference_start_ms) / 1000.0;
                    num_prompt_tokens = static_cast<int>(s.num_prompt_tokens);
                    num_generated_tokens = static_cast<int>(s.num_generated_tokens);
                }
        
                output_file << "{";
                output_file << "\"id\": " << row_index << ", ";
                output_file << "\"prompt\": \"" << escape_json(prompt) << "\", ";
                output_file << "\"model_output\": \"" << escape_json(model_output) << "\", ";
                output_file << "\"logs\": \"\", ";
                output_file << "\"timings\": {";
                output_file << "\"model_load_s\": " << std::fixed << std::setprecision(6) << model_load_s << ", ";
                output_file << "\"inference_s\": " << std::fixed << std::setprecision(6) << inference_s << ", ";
                output_file << "\"first_token_s\": " << std::fixed << std::setprecision(6) << first_token_s << ", ";
                output_file << "\"num_prompt_tokens\": " << num_prompt_tokens << ", ";
                output_file << "\"num_generated_tokens\": " << num_generated_tokens;
                output_file << "}, ";
                output_file << "\"system_metrics\": {\"rss_mb\": null}, ";
                output_file << "\"wallclock_s\": " << std::fixed << std::setprecision(6) << wallclock_s.count();
                output_file << "}\n";
            }
        }
        

        else if (dataset_type == "scientific_mcq") {
            // --- Use the same robust flexible CSV parsing from MMLU ---
            std::vector<std::string> headers;
            std::vector<std::vector<std::string>> rows;
            
            // This assumes the read_csv_rows_multiline function is defined elsewhere in your file
            if (!read_csv_rows_multiline(csv_path, headers, rows)) {
                ET_LOG(Error, "Failed to parse CSV for scientific_mcq: %s", csv_path.c_str());
                return;
            }

            // Normalize header names (same as MMLU)
            for (auto &h : headers) {
                std::string hh = h;
                std::transform(hh.begin(), hh.end(), hh.begin(), ::tolower);
                h = hh;
            }

            // Find column function (same as MMLU)
            auto find_col = [&](const std::string &name) -> int {
                for (size_t i = 0; i < headers.size(); ++i) {
                    std::string h = headers[i];
                    // remove quotes and trim
                    if (!h.empty() && h.front() == '"' && h.back() == '"') {
                        h = h.substr(1, h.size()-2);
                    }
                    size_t s = h.find_first_not_of(" \t");
                    size_t e = h.find_last_not_of(" \t");
                    if (s != std::string::npos && e != std::string::npos) {
                        h = h.substr(s, e-s+1);
                    }
                    std::string lower = h;
                    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                    if (lower == name) return static_cast<int>(i);
                }
                return -1;
            };

            // --- Find the required columns for scientific_mcq ---
            int idx_q = find_col("question");
            int idx_d1 = find_col("distractor1");
            int idx_d2 = find_col("distractor2");
            int idx_d3 = find_col("distractor3");
            int idx_ans = find_col("correct_answer");
            // Note: 'support' is not used in the prompt, so we don't need to find it

            // Check that all required columns were found
            if (idx_q < 0 || idx_d1 < 0 || idx_d2 < 0 || idx_d3 < 0 || idx_ans < 0) {
                ET_LOG(Error, "CSV does not match expected scientific_mcq headers.");
                if (idx_q < 0) ET_LOG(Error, "Missing 'question' column.");
                if (idx_d1 < 0) ET_LOG(Error, "Missing 'distractor1' column.");
                if (idx_d2 < 0) ET_LOG(Error, "Missing 'distractor2' column.");
                if (idx_d3 < 0) ET_LOG(Error, "Missing 'distractor3' column.");
                if (idx_ans < 0) ET_LOG(Error, "Missing 'correct_answer' column.");
                return;
            }
            
            // --- Iterate over parsed rows (same as MMLU) ---
            for (const auto& cols : rows) {
                int max_idx = std::max({idx_q, idx_d1, idx_d2, idx_d3, idx_ans});
                if (static_cast<int>(cols.size()) <= max_idx) {
                    ET_LOG(Info, "Skipping row with too few columns (got %zu, need at least %d)", cols.size(), max_idx + 1);
                    continue;
                }

                // --- Get the data from the correct columns by index ---
                std::string q = cols[idx_q];
                std::string d1 = cols[idx_d1];
                std::string d2 = cols[idx_d2];
                std::string d3 = cols[idx_d3];
                std::string ans = cols[idx_ans];

                // --- Build the original prompt ---
                std::string prompt = "Choose the correct option for the given question.\n\nQuestion: " + q +
                                     "\n\nOptions:\n1) " + d1 + "\n2) " + d2 + "\n3) " + d3 + "\n4) " + ans +
                                     "\n\nAnswer with only the option option (1/2/3/4).";

               // --- All the remaining code (runner, stats, JSON output) is identical ---
                
                // This is the log for every prompt
                std::cout << "Processing entry " << ++row_index << "..." << std::endl;
                runner->reset();
                ::executorch::extension::llm::GenerationConfig config {
                    .max_new_tokens = static_cast<int32_t>(FLAGS_seq_len),
                    .temperature = static_cast<float>(FLAGS_temperature),
                };
                std::stringstream model_output_stream;
                auto token_callback = [&](const std::string& piece) { model_output_stream << piece; };

                std::optional<::executorch::extension::llm::Stats> stats_snapshot;
                auto stats_callback = [&](const ::executorch::extension::llm::Stats& s) {
                    stats_snapshot.emplace(s);
                };

                auto start_time = std::chrono::high_resolution_clock::now();
                auto err = runner->generate(prompt, config, token_callback, stats_callback);
                auto end_time = std::chrono::high_resolution_clock::now();

                if (err != executorch::runtime::Error::Ok) {
                    ET_LOG(Error, "Inference failed for entry %d", row_index);
                    double wallclock_s = std::chrono::duration<double>(end_time - start_time).count();
                    output_file << "{\"id\": " << row_index
                                << ", \"prompt\": \"" << escape_json(prompt)
                                << "\", \"model_output\": \"\", \"logs\": \"\", \"timings\": {}, \"system_metrics\": {}, \"wallclock_s\": " << wallclock_s << "}" << std::endl; // <-- FIX HERE
                    continue;
                }

                std::string model_output = model_output_stream.str();
                std::chrono::duration<double> wallclock_s = end_time - start_time;

                double model_load_s = -1.0;
                double inference_s = -1.0;
                double first_token_s = -1.0;
                int num_prompt_tokens = -1;
                int num_generated_tokens = -1;
                if (stats_snapshot.has_value()) {
                    auto &s = stats_snapshot.value();
                    model_load_s = (s.model_load_end_ms - s.model_load_start_ms) / 1000.0;
                    inference_s = (s.inference_end_ms - s.inference_start_ms) / 1000.0;
                    first_token_s = (s.first_token_ms - s.inference_start_ms) / 1000.0;
                    num_prompt_tokens = static_cast<int>(s.num_prompt_tokens);
                    num_generated_tokens = static_cast<int>(s.num_generated_tokens);
                }

                output_file << "{";
                output_file << "\"id\": " << row_index << ", ";
                output_file << "\"prompt\": \"" << escape_json(prompt) << "\", ";
                output_file << "\"model_output\": \"" << escape_json(model_output) << "\", ";
                output_file << "\"logs\": \"\", ";
                output_file << "\"timings\": {";
                output_file << "\"model_load_s\": " << std::fixed << std::setprecision(6) << model_load_s << ", ";
                output_file << "\"inference_s\": " << std::fixed << std::setprecision(6) << inference_s << ", ";
                output_file << "\"first_token_s\": " << std::fixed << std::setprecision(6) << first_token_s << ", ";
                output_file << "\"num_prompt_tokens\": " << num_prompt_tokens << ", ";
                output_file << "\"num_generated_tokens\": " << num_generated_tokens;
                output_file << "}, ";
                output_file << "\"system_metrics\": {\"rss_mb\": null}, ";
                output_file << "\"wallclock_s\": " << std::fixed << std::setprecision(6) << wallclock_s.count();
                output_file << "}" << std::endl; // <-- FIX HERE
            }

            // --- This is the final log after all entries are processed ---
            std::cout << "Finished processing " << row_index << " entries." << std::endl;
            std::cout << "Results saved to: " << FLAGS_output_json << std::endl;
            // --- End of final log ---

         }else if (dataset_type == "mmlu_mcq") {
            // --- Robust flexible CSV parsing for MMLU (handles multi-line and quoted fields) ---
            std::vector<std::string> headers;
            std::vector<std::vector<std::string>> rows;
            if (!read_csv_rows_multiline(csv_path, headers, rows)) {
                ET_LOG(Error, "Failed to parse CSV for MMLU: %s", csv_path.c_str());
                return;
            }

            // Normalize header names to lowercase trimmed
            for (auto &h : headers) {
                std::string hh = h;
                std::transform(hh.begin(), hh.end(), hh.begin(), ::tolower);
                h = hh;
            }

            // Find required column indices
            auto find_col = [&](const std::string &name) -> int {
                for (size_t i = 0; i < headers.size(); ++i) {
                    std::string h = headers[i];
                    // remove quotes and trim
                    if (!h.empty() && h.front() == '"' && h.back() == '"') {
                        h = h.substr(1, h.size()-2);
                    }
                    size_t s = h.find_first_not_of(" \t");
                    size_t e = h.find_last_not_of(" \t");
                    if (s != std::string::npos && e != std::string::npos) {
                        h = h.substr(s, e-s+1);
                    }
                    std::string lower = h;
                    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                    if (lower == name) return static_cast<int>(i);
                }
                return -1;
            };

            int idx_q = find_col("question");
            int idx_subject = find_col("subject");
            int idx_choices = find_col("choices");
            int idx_options = find_col("options"); // fallback
            int idx_answer = find_col("answer");

            if (idx_q < 0 || idx_answer < 0 || (idx_choices < 0 && idx_options < 0)) {
                ET_LOG(Error, "CSV does not match expected MMLU headers (need question & choices/options & answer).");
                return;
            }

            // iterate over parsed rows
            for (const auto& cols : rows) {
                // skip rows that have fewer columns than header (rare but possible)
                if (static_cast<int>(cols.size()) <= std::max({idx_q, idx_choices >= 0 ? idx_choices : idx_options, idx_answer})) {
                    // log and continue
                    ET_LOG(Info, "Skipping row with too few columns (got %zu, header expects %zu)", cols.size(), headers.size());
                    continue;
                }

                std::string q = cols[idx_q];
                std::string choices_field = (idx_choices >= 0) ? cols[idx_choices] : cols[idx_options];
                // Use regex to find all single-quoted choices: '...'
                std::vector<std::string> choice_list;
                std::regex re("'([^']*)'");
                std::smatch m;
                std::string tmp = choices_field;
                auto begin = std::sregex_iterator(tmp.begin(), tmp.end(), re);
                auto end = std::sregex_iterator();
                for (auto it = begin; it != end; ++it) {
                    std::smatch match = *it;
                    if (match.size() >= 2) {
                        choice_list.push_back(match[1].str());
                    }
                }

                // Fallback: if no single-quoted items found, remove brackets/quotes and split on commas
                if (choice_list.empty()) {
                    std::string c = choices_field;
                    // remove surrounding brackets and single quotes
                    c.erase(std::remove(c.begin(), c.end(), '\''), c.end());
                    if (!c.empty() && (c.front() == '[' || c.front() == '(')) c.erase(c.begin());
                    if (!c.empty() && (c.back() == ']' || c.back() == ')')) c.pop_back();
                    std::stringstream ss(c);
                    std::string choice;
                    while (std::getline(ss, choice, ',')) {
                        // trim
                        size_t s = choice.find_first_not_of(" \t");
                        size_t e = choice.find_last_not_of(" \t");
                        if (s != std::string::npos && e != std::string::npos) {
                            choice_list.push_back(choice.substr(s, e - s + 1));
                        } else if (!choice.empty()) {
                            choice_list.push_back(choice);
                        }
                    }
                }

                if (choice_list.size() < 4) {
                   ET_LOG(Info, "Skipping MMLU row: fewer than 4 choices detected for question: %s", q.c_str());
                    continue;
                }

                std::string prompt = "Choose the correct option for the given question.\n\nQuestion: " + q +
                                     "\n\nOptions:\n1) " + choice_list[0] + "\n2) " + choice_list[1] +
                                     "\n3) " + choice_list[2] + "\n4) " + choice_list[3] +
                                     "\n\nAnswer with only the option number (1/2/3/4).";

                std::cout << "Processing entry " << ++row_index << "..." << std::endl;
                runner->reset();
                ::executorch::extension::llm::GenerationConfig config{
                    .max_new_tokens = static_cast<int32_t>(FLAGS_seq_len),
                    .temperature = static_cast<float>(FLAGS_temperature),
                };
                std::stringstream model_output_stream;
                auto token_callback = [&](const std::string& piece) { model_output_stream << piece; };

                std::optional<::executorch::extension::llm::Stats> stats_snapshot;
                auto stats_callback = [&](const ::executorch::extension::llm::Stats& s) {
                    stats_snapshot.emplace(s);
                };

                auto start_time = std::chrono::high_resolution_clock::now();
                auto err = runner->generate(prompt, config, token_callback, stats_callback);
                auto end_time = std::chrono::high_resolution_clock::now();

                if (err != executorch::runtime::Error::Ok) {
                    ET_LOG(Error, "Inference failed for entry %d", row_index);
                    double wallclock_s = std::chrono::duration<double>(end_time - start_time).count();
                    output_file << "{\"id\": " << row_index
                                << ", \"prompt\": \"" << escape_json(prompt)
                                << "\", \"model_output\": \"\", \"logs\": \"\", \"timings\": {}, \"system_metrics\": {}, \"wallclock_s\": " << wallclock_s << "}" << std::endl; // <-- FIX HERE
                    continue;
                }

                std::string model_output = model_output_stream.str();
                std::chrono::duration<double> wallclock_s = end_time - start_time;

                double model_load_s = -1.0;
                double inference_s = -1.0;
                double first_token_s = -1.0;
                int num_prompt_tokens = -1;
                int num_generated_tokens = -1;
                if (stats_snapshot.has_value()) {
                    auto &s = stats_snapshot.value();
                    model_load_s =
                        (s.model_load_end_ms - s.model_load_start_ms) / 1000.0;
                    inference_s =
                        (s.inference_end_ms - s.inference_start_ms) / 1000.0;
                    first_token_s =
                        (s.first_token_ms - s.inference_start_ms) / 1000.0;
                    num_prompt_tokens = static_cast<int>(s.num_prompt_tokens);
                    num_generated_tokens = static_cast<int>(s.num_generated_tokens);
                }

                output_file << "{";
                output_file << "\"id\": " << row_index << ", ";
                output_file << "\"prompt\": \"" << escape_json(prompt) << "\", ";
                output_file << "\"model_output\": \"" << escape_json(model_output) << "\", ";
                output_file << "\"logs\": \"\", ";
                output_file << "\"timings\": {";
                output_file << "\"model_load_s\": " << std::fixed << std::setprecision(6)
                            << model_load_s << ", ";
                output_file << "\"inference_s\": " << std::fixed << std::setprecision(6)
                            << inference_s << ", ";
                output_file << "\"first_token_s\": " << std::fixed << std::setprecision(6)
                            << first_token_s << ", ";
                output_file << "\"num_prompt_tokens\": " << num_prompt_tokens << ", ";
                output_file << "\"num_generated_tokens\": " << num_generated_tokens;
                output_file << "}, ";
                output_file << "\"system_metrics\": {\"rss_mb\": null}, ";
                output_file << "\"wallclock_s\": " << std::fixed << std::setprecision(6)
                            << wallclock_s.count();
                output_file << "}" << std::endl; // <-- FIX HERE
            }
        } else {
             ET_LOG(Error, "Unknown or unsupported dataset type: %s", dataset_type.c_str());
             return;
         }
     } catch (const std::exception& e) {
         ET_LOG(Error, "Error processing CSV file %s: %s", csv_path.c_str(), e.what());
     }
 }

 int32_t main(int32_t argc, char** argv) {
     gflags::ParseCommandLineFlags(&argc, &argv, true);

     if (FLAGS_dataset_type.empty() || FLAGS_csv_path.empty()) {
         ET_LOG(Error, "Error: --dataset_type and --csv_path are required for evaluation.");
         return 1;
     }

     const char* model_path = FLAGS_model_path.c_str();
     std::vector<std::string> data_paths = parseStringList(FLAGS_data_paths);
     const char* tokenizer_path = FLAGS_tokenizer_path.c_str();

     std::unique_ptr<::executorch::extension::llm::TextLLMRunner> runner =
         example::create_llama_runner(model_path, tokenizer_path, data_paths);

     if (runner == nullptr) {
         ET_LOG(Error, "Failed to create llama runner");
         return 1;
     }

     std::ofstream output_file(FLAGS_output_json);
     if (!output_file.is_open()) {
         ET_LOG(Error, "Failed to open output file: %s", FLAGS_output_json.c_str());
         return 1;
     }

     ET_LOG(Info, "Starting evaluation on dataset: %s", FLAGS_csv_path.c_str());
     run_evaluation(runner.get(), FLAGS_dataset_type, FLAGS_csv_path, output_file);

     output_file.close();
     ET_LOG(Info, "Evaluation finished. Results saved to %s", FLAGS_output_json.c_str());

     return 0;
 }
