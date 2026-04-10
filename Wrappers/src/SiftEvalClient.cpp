// SiftEvalClient.cpp
// Multi-threaded C++ evaluation client using AnnClient against SIFT1M.
//
// Usage (example):
//   ./sift_eval_client \
//       --query_path   /home/akchunks/Desktop/createlab/data/sift1M/sift_query.fvecs \
//       --gt_path      /home/akunte2/work/horizann/data/sift1M/sift_groundtruth.ivecs \
//       --host         127.0.0.1 \
//       --port         9200 \
//       --K            10 \
//       --max_queries  10000 \
//       --num_threads  8
//
// Build (example):
//   g++ -O3 -std=c++17 SiftEvalClient.cpp -o sift_eval_client -pthread \
//       -I/path/to/sptag/include -L/path/to/sptag/lib -lSPTAGClient

#include "inc/ClientInterface.h"      // AnnClient
#include "inc/TransferDataType.h"     // ByteArray, BasicResult, RemoteSearchResult typedefs

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <thread>
#include <cctype>

// ---------------------- CLI args ---------------------- //

struct Args {
    std::string query_path =
        "/home/akunte2/work/horizann/data/sift1M/sift_query.fvecs";
        
    std::string gt_path =
        "/home/akunte2/work/horizann/data/sift1M/sift_groundtruth.ivecs";
    std::string host = "127.0.0.1";
    std::string value_type = "Float";
    int port = 9200;
    int K = 10;
    int max_queries = -1;   // -1 = all
    int num_threads = 8;    // number of worker threads
};

void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "  --query_path   PATH  (default: "
        << "/home/akunte2/work/horizann/data/sift1M/sift_query.fvecs)\n"
        << "  --gt_path      PATH  (default: "
        << "/home/akunte2/work/horizann/data/sift1M/sift_groundtruth.ivecs)\n"
        << "  --host         HOST  (default: 127.0.0.1)\n"
        << "  --value_type   TYPE  (default: Float)\n"
        << "  --port         PORT  (default: 9200)\n"
        << "  --K            K     (default: 10)\n"
        << "  --max_queries  N     (default: all)\n"
        << "  --num_threads  T     (default: 8)\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key(argv[i]);
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return std::string(argv[++i]);
        };

        if (key == "--query_path") {
            args.query_path = need_value("--query_path");
        } else if (key == "--gt_path") {
            args.gt_path = need_value("--gt_path");
        } else if (key == "--host") {
            args.host = need_value("--host");
        } else if (key == "--value_type") {
            args.value_type = need_value("--value_type");
        } else if (key == "--port") {
            args.port = std::stoi(need_value("--port"));
        } else if (key == "--K") {
            args.K = std::stoi(need_value("--K"));
        } else if (key == "--max_queries") {
            args.max_queries = std::stoi(need_value("--max_queries"));
        } else if (key == "--num_threads") {
            args.num_threads = std::stoi(need_value("--num_threads"));
        } else if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << key << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    if (args.num_threads <= 0) {
        args.num_threads = 1;
    }
    return args;
}

// ---------------------- .fvecs / .ivecs readers ---------------------- //
//
// Faiss-style records:
//   int32 dim
//   dim * value_type
//

void read_fvecs(const std::string& path,
                std::vector<float>& out,
                std::int32_t& dim,
                std::int32_t& n) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Failed to open fvecs file: " + path);
    }

    fin.seekg(0, std::ios::end);
    std::streamoff sz = fin.tellg();
    fin.seekg(0, std::ios::beg);
    if (sz <= 0) {
        throw std::runtime_error("Empty fvecs file: " + path);
    }

    std::int32_t dim_tmp = 0;
    fin.read(reinterpret_cast<char*>(&dim_tmp), sizeof(std::int32_t));
    if (!fin) {
        throw std::runtime_error("Failed to read header from fvecs file: " + path);
    }
    if (dim_tmp <= 0) {
        throw std::runtime_error("Invalid dim in fvecs: " + std::to_string(dim_tmp));
    }
    dim = dim_tmp;

    const std::int64_t bytes_per_vec =
        sizeof(std::int32_t) + static_cast<std::int64_t>(dim) * sizeof(float);
    const std::int64_t total_bytes = static_cast<std::int64_t>(sz);
    if (total_bytes % bytes_per_vec != 0) {
        throw std::runtime_error("File size not divisible by vector record size in fvecs: " + path);
    }
    n = static_cast<std::int32_t>(total_bytes / bytes_per_vec);

    fin.clear();
    fin.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<std::size_t>(total_bytes));
    fin.read(buf.data(), total_bytes);
    if (!fin) {
        throw std::runtime_error("Failed to read full fvecs file: " + path);
    }

    out.resize(static_cast<std::size_t>(n) * dim);
    const char* p = buf.data();
    for (int i = 0; i < n; ++i) {
        const std::int32_t* dim_ptr = reinterpret_cast<const std::int32_t*>(p);
        if (*dim_ptr != dim) {
            throw std::runtime_error("Inconsistent dim in fvecs row");
        }
        p += sizeof(std::int32_t);
        const float* vals = reinterpret_cast<const float*>(p);
        std::memcpy(&out[static_cast<std::size_t>(i) * dim],
                    vals,
                    static_cast<std::size_t>(dim) * sizeof(float));
        p += static_cast<std::size_t>(dim) * sizeof(float);
    }
}

void read_bvecs(const std::string& path,
                std::vector<float>& out,
                std::int32_t& dim,
                std::int32_t& n) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Failed to open bvecs file: " + path);
    }

    fin.seekg(0, std::ios::end);
    std::streamoff sz = fin.tellg();
    fin.seekg(0, std::ios::beg);
    if (sz <= 0) {
        throw std::runtime_error("Empty bvecs file: " + path);
    }

    std::int32_t dim_tmp = 0;
    fin.read(reinterpret_cast<char*>(&dim_tmp), sizeof(std::int32_t));
    if (!fin) {
        throw std::runtime_error("Failed to read header from bvecs file: " + path);
    }
    if (dim_tmp <= 0) {
        throw std::runtime_error("Invalid dim in bvecs: " + std::to_string(dim_tmp));
    }
    dim = dim_tmp;

    const std::int64_t bytes_per_vec =
        sizeof(std::int32_t) + static_cast<std::int64_t>(dim) * sizeof(std::uint8_t);
    const std::int64_t total_bytes = static_cast<std::int64_t>(sz);
    if (total_bytes % bytes_per_vec != 0) {
        throw std::runtime_error("File size not divisible by vector record size in bvecs: " + path);
    }
    n = static_cast<std::int32_t>(total_bytes / bytes_per_vec);

    fin.clear();
    fin.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<std::size_t>(total_bytes));
    fin.read(buf.data(), total_bytes);
    if (!fin) {
        throw std::runtime_error("Failed to read full bvecs file: " + path);
    }

    out.resize(static_cast<std::size_t>(n) * dim);
    const char* p = buf.data();
    for (int i = 0; i < n; ++i) {
        const std::int32_t* dim_ptr = reinterpret_cast<const std::int32_t*>(p);
        if (*dim_ptr != dim) {
            throw std::runtime_error("Inconsistent dim in bvecs row");
        }
        p += sizeof(std::int32_t);
        const std::uint8_t* vals = reinterpret_cast<const std::uint8_t*>(p);
        float* out_row = &out[static_cast<std::size_t>(i) * dim];
        for (int j = 0; j < dim; ++j) {
            out_row[j] = static_cast<float>(vals[j]);
        }
        p += static_cast<std::size_t>(dim) * sizeof(std::uint8_t);
    }
}

void read_ivecs(const std::string& path,
                std::vector<std::int32_t>& out,
                std::int32_t& gt_dim,
                std::int32_t& n) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Failed to open ivecs file: " + path);
    }

    fin.seekg(0, std::ios::end);
    std::streamoff sz = fin.tellg();
    fin.seekg(0, std::ios::beg);
    if (sz <= 0) {
        throw std::runtime_error("Empty ivecs file: " + path);
    }

    std::int32_t dim_tmp = 0;
    fin.read(reinterpret_cast<char*>(&dim_tmp), sizeof(std::int32_t));
    if (!fin) {
        throw std::runtime_error("Failed to read header from ivecs file: " + path);
    }
    if (dim_tmp <= 0) {
        throw std::runtime_error("Invalid dim in ivecs: " + std::to_string(dim_tmp));
    }
    gt_dim = dim_tmp;

    const std::int64_t bytes_per_vec =
        sizeof(std::int32_t) + static_cast<std::int64_t>(gt_dim) * sizeof(std::int32_t);
    const std::int64_t total_bytes = static_cast<std::int64_t>(sz);
    if (total_bytes % bytes_per_vec != 0) {
        throw std::runtime_error("File size not divisible by vector record size in ivecs: " + path);
    }
    n = static_cast<std::int32_t>(total_bytes / bytes_per_vec);

    fin.clear();
    fin.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<std::size_t>(total_bytes));
    fin.read(buf.data(), total_bytes);
    if (!fin) {
        throw std::runtime_error("Failed to read full ivecs file: " + path);
    }

    out.resize(static_cast<std::size_t>(n) * gt_dim);
    const char* p = buf.data();
    for (int i = 0; i < n; ++i) {
        const std::int32_t* dim_ptr = reinterpret_cast<const std::int32_t*>(p);
        if (*dim_ptr != gt_dim) {
            throw std::runtime_error("Inconsistent dim in ivecs row");
        }
        p += sizeof(std::int32_t);
        const std::int32_t* vals = reinterpret_cast<const std::int32_t*>(p);
        std::memcpy(&out[static_cast<std::size_t>(i) * gt_dim],
                    vals,
                    static_cast<std::size_t>(gt_dim) * sizeof(std::int32_t));
        p += static_cast<std::size_t>(gt_dim) * sizeof(std::int32_t);
    }
}

// ---------------------- Recall@K ---------------------- //

double compute_recall_at_k(const std::vector<std::int64_t>& pred_ids, // [Q * K]
                           const std::vector<std::int32_t>& gt_ids,   // [Q * gt_dim]
                           int Q,
                           int K,
                           int gt_dim) {
    double total = 0.0;
    for (int q = 0; q < Q; ++q) {
        const std::int64_t* pred_row = &pred_ids[static_cast<std::size_t>(q) * K];
        const std::int32_t* gt_row   = &gt_ids[static_cast<std::size_t>(q) * gt_dim];

        std::unordered_set<std::int32_t> gt_set;
        gt_set.reserve(K * 2);
        for (int i = 0; i < K; ++i) {
            gt_set.insert(gt_row[i]);
        }

        int hit = 0;
        for (int i = 0; i < K; ++i) {
            if (gt_set.find(static_cast<std::int32_t>(pred_row[i])) != gt_set.end()) {
                ++hit;
            }
        }

        total += static_cast<double>(hit) / static_cast<double>(K);
    }
    return total / static_cast<double>(Q);
}

// ---------------------- Helpers ---------------------- //

// Trim whitespace from both ends
static std::string trim(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

static std::string lowercase(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

static std::string get_extension_lower(const std::string& path) {
    std::size_t pos = path.find_last_of('.');
    if (pos == std::string::npos) return "";
    return lowercase(path.substr(pos));
}

// ---------------------- Worker function ---------------------- //
//
// Each thread:
//
//  - creates its own AnnClient
//  - waits until connected
//  - processes queries in [q_start, q_end)
//  - writes results into pred_ids / pred_dists at disjoint ranges
//

void search_range(const Args& args,
                  int q_start,
                  int q_end,
                  const std::vector<float>& queries,
                  int dim,
                  int K,
                  std::vector<std::int64_t>& pred_ids,
                  std::vector<double>& pred_dists)
{
    if (q_start >= q_end) return;

    const std::string value_type = lowercase(args.value_type);
    if (value_type != "float" && value_type != "uint8") {
        throw std::runtime_error("Unsupported --value_type: " + args.value_type);
    }

    std::string port_str = std::to_string(args.port);
    AnnClient client(args.host.c_str(), port_str.c_str());
    std::vector<std::uint8_t> query_u8(static_cast<std::size_t>(dim));

    while (!client.IsConnected()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    client.SetTimeoutMilliseconds(20000);

    for (int q = q_start; q < q_end; ++q) {
        const float* qptr = &queries[static_cast<std::size_t>(q) * dim];

        ByteArray ba;
        const char* request_value_type = nullptr;
        if (value_type == "uint8") {
            for (int i = 0; i < dim; ++i) {
                float v = std::round(qptr[i]);
                if (v < 0.0f) v = 0.0f;
                if (v > 255.0f) v = 255.0f;
                query_u8[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(v);
            }
            ba = ByteArray(query_u8.data(), static_cast<std::size_t>(dim), false);
            request_value_type = "UInt8";
        } else {
            ba = ByteArray(
                reinterpret_cast<std::uint8_t*>(const_cast<float*>(qptr)),
                static_cast<std::size_t>(dim) * sizeof(float),
                false  // do not free qptr
            );
            request_value_type = "Float";
        }

        std::shared_ptr<RemoteSearchResult> res_ptr =
            client.Search(ba, K, request_value_type, true);

        if (!res_ptr) {
            throw std::runtime_error("AnnClient::Search returned null result for q=" + std::to_string(q));
        }

        RemoteSearchResult& rs = *res_ptr;
        if (rs.m_status != SPTAG::Socket::RemoteSearchResult::ResultStatus::Success) {
            throw std::runtime_error("Search failed or timed out for query " + std::to_string(q));
        }

        // Flatten all results from all index results
        std::vector<std::int64_t> all_ids;
        std::vector<float> all_dists;

        for (const auto& indexRes : rs.m_allIndexResults) {
            const QueryResult& qr = indexRes.m_results;
            const int rnum = qr.GetResultNum();
            for (int i = 0; i < rnum; ++i) {
                const BasicResult* br = qr.GetResult(i);
                if (!br) continue;

                std::int64_t gid = -1;
                const ByteArray& meta = br->Meta;
                if (meta.Length() > 0) {
                    std::string s(reinterpret_cast<const char*>(meta.Data()),
                                  meta.Length());
                    s = trim(s);
                    try {
                        gid = static_cast<std::int64_t>(std::stoll(s));
                    } catch (...) {
                        gid = static_cast<std::int64_t>(br->VID);
                    }
                } else {
                    gid = static_cast<std::int64_t>(br->VID);
                }

                all_ids.push_back(gid);
                all_dists.push_back(br->Dist);
            }
        }

        if (all_ids.size() < static_cast<std::size_t>(K)) {
            throw std::runtime_error(
                "Search returned only " + std::to_string(all_ids.size()) +
                " neighbors (< K=" + std::to_string(K) + ") for query " +
                std::to_string(q));
        }

        // Sort by distance, take top K
        std::vector<std::size_t> order(all_ids.size());
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + K, order.end(),
                          [&](std::size_t a, std::size_t b) {
                              return all_dists[a] < all_dists[b];
                          });

        for (int j = 0; j < K; ++j) {
            std::size_t idx = order[j];
            pred_ids[static_cast<std::size_t>(q) * K + j] =
                all_ids[idx];
            pred_dists[static_cast<std::size_t>(q) * K + j] =
                static_cast<double>(all_dists[idx]);
        }

        // Optional: per-thread logging
        // if ((q - q_start + 1) % 1000 == 0) {
        //     std::cout << "[Thread " << std::this_thread::get_id()
        //               << "] processed " << (q - q_start + 1)
        //               << " queries (global q=" << q << ")\n";
        // }
    }
}

// ---------------------- Main ---------------------- //

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        // std::cout << "Loading queries from:    " << args.query_path << "\n";
        // std::cout << "Loading groundtruth from:" << args.gt_path << "\n";

        std::vector<float> queries;
        std::vector<std::int32_t> gt_all;
        std::int32_t dim = 0, n_queries = 0;
        std::int32_t gt_dim = 0, n_gt = 0;

        const std::string q_ext = get_extension_lower(args.query_path);
        if (q_ext == ".fvecs") {
            read_fvecs(args.query_path, queries, dim, n_queries);
        } else if (q_ext == ".bvecs") {
            read_bvecs(args.query_path, queries, dim, n_queries);
        } else {
            throw std::runtime_error("Unsupported query file extension: " + q_ext +
                                     " (expected .fvecs or .bvecs)");
        }
        read_ivecs(args.gt_path, gt_all, gt_dim, n_gt);

        if (n_queries != n_gt) {
            std::cerr << "Warning: query count (" << n_queries
                      << ") != gt count (" << n_gt << ")\n";
        }

        int Q = std::min(n_queries, n_gt);
        if (args.max_queries > 0 && args.max_queries < Q) {
            Q = args.max_queries;
        }

        if (args.K > gt_dim) {
            throw std::runtime_error("K > groundtruth dimension");
        }

        if (Q <= 0) {
            throw std::runtime_error("No queries to process (Q <= 0).");
        }

        std::cout << Q << " queries, dim=" << dim
                  << ", gt_dim=" << gt_dim
                  << ", K=" << args.K
                  << ", value_type=" << args.value_type
                  << ", num_threads=" << args.num_threads
                  << "\n";

        const int K = args.K;
        std::vector<std::int64_t> pred_ids(static_cast<std::size_t>(Q) * K, -1);
        std::vector<double>       pred_dists(static_cast<std::size_t>(Q) * K, 0.0);

        auto t0 = std::chrono::steady_clock::now();

        // --- Launch threads, partition queries [0, Q) --- //
        int num_threads = std::min(args.num_threads, Q);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        int base_chunk = Q / num_threads;
        int remainder = Q % num_threads;
        int q_start = 0;

        for (int t = 0; t < num_threads; ++t) {
            int chunk_size = base_chunk + (t < remainder ? 1 : 0);
            int q_end = q_start + chunk_size;

            threads.emplace_back(
                [&, q_start, q_end]() {
                    // Each thread uses its own AnnClient internally
                    search_range(args, q_start, q_end,
                                 queries, dim, K,
                                 pred_ids, pred_dists);
                }
            );

            q_start = q_end;
        }

        for (auto& th : threads) {
            th.join();
        }

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double qps = (elapsed > 0.0) ? (static_cast<double>(Q) / elapsed) : 0.0;

        // Compute Recall@K
        double recall = compute_recall_at_k(pred_ids, gt_all, Q, K, gt_dim);

        double sum_dist = 0.0;
        for (double d : pred_dists) sum_dist += d;
        double mean_dist = sum_dist / static_cast<double>(Q * K);

        std::cout << "\nRecall@" << K << ": " << recall << "\n";
        std::cout << "Mean distance: " << mean_dist << "\n";
        std::cout << "Total time: " << elapsed << " s, QPS: " << qps << "\n";

        // First-query debug
        // std::cout << "\nFirst query debug:\n";
        // std::cout << "Groundtruth IDs (top K): ";
        // for (int i = 0; i < K; ++i) {
        //     std::cout << gt_all[i] << (i + 1 < K ? " " : "\n");
        // }

        // std::cout << "Predicted IDs:          ";
        // for (int i = 0; i < K; ++i) {
        //     std::cout << pred_ids[i] << (i + 1 < K ? " " : "\n");
        // }

        // std::cout << "Predicted distances:    ";
        // for (int i = 0; i < K; ++i) {
        //     std::cout << pred_dists[i] << (i + 1 < K ? " " : "\n");
        // }

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << "\n";
        return 1;
    }
}
