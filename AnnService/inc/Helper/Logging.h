// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_LOGGING_H_
#define _SPTAG_HELPER_LOGGING_H_

#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <fstream>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>

#pragma warning(disable:4996)

namespace SPTAG
{
    namespace Helper
    {
        enum class LogLevel
        {
            LL_Debug = 0,
            LL_Info,
            LL_Status,
            LL_Warning,
            LL_Error,
            LL_Assert,
            LL_Count,
            LL_Empty
        };

        inline const char* LogLevelName(LogLevel level)
        {
            switch (level)
            {
            case LogLevel::LL_Debug: return "DEBUG";
            case LogLevel::LL_Info: return "INFO";
            case LogLevel::LL_Status: return "STATUS";
            case LogLevel::LL_Warning: return "WARN";
            case LogLevel::LL_Error: return "ERROR";
            case LogLevel::LL_Assert: return "ASSERT";
            default: return "LOG";
            }
        }

        inline int GetNumericThreadId()
        {
            static std::mutex s_threadMapLock;
            static std::unordered_map<std::thread::id, int> s_threadIdMap;
            static std::atomic<int> s_nextThreadId(1);

            const std::thread::id tid = std::this_thread::get_id();
            std::lock_guard<std::mutex> lock(s_threadMapLock);
            auto it = s_threadIdMap.find(tid);
            if (it != s_threadIdMap.end())
            {
                return it->second;
            }

            int assigned = s_nextThreadId.fetch_add(1, std::memory_order_relaxed);
            s_threadIdMap.emplace(tid, assigned);
            return assigned;
        }

        class Logger 
        {
        public:
            virtual void Logging(const char* title, LogLevel level, const char* file, int line, const char* func, const char* format, ...) = 0;
        };

        class LoggerHolder
        {
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 	202002L) || __cplusplus >= 	202002L)
        private:
            std::atomic<std::shared_ptr<Logger>> m_logger;
        public:
            LoggerHolder(std::shared_ptr<Logger> logger) : m_logger(logger) {}

            void SetLogger(std::shared_ptr<Logger> p_logger)
            {
                m_logger = p_logger;
            }

            std::shared_ptr<Logger> GetLogger()
            {
                return m_logger;
            }
#else
        private:
            std::shared_ptr<Logger> m_logger;
        public:
            LoggerHolder(std::shared_ptr<Logger> logger) : m_logger(logger) {}

            void SetLogger(std::shared_ptr<Logger> p_logger)
            {
                std::atomic_store(&m_logger, p_logger);
            }

            std::shared_ptr<Logger> GetLogger()
            {
                return std::atomic_load(&m_logger);
            }
#endif
        };


        class SimpleLogger : public Logger {
        public:
            SimpleLogger(LogLevel level) : m_level(level), m_start(std::chrono::steady_clock::now()) {}

            virtual void Logging(const char* title, LogLevel level, const char* file, int line, const char* func, const char* format, ...)
            {
                if (level < m_level) return;

                va_list args;
                va_start(args, format);
                char buffer[4096];
                int ret = vsnprintf(buffer, sizeof(buffer), format, args);
                va_end(args);
                if (ret < 0) return;

                auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - m_start)
                                     .count();
                long long sec = elapsedMs / 1000;
                long long ms = elapsedMs % 1000;
                int numericThreadId = GetNumericThreadId();

                std::lock_guard<std::mutex> lock(m_lock);
                if (level != LogLevel::LL_Empty)
                {
                    printf("[+%lld.%03llds][%s][thread %d] ", sec, ms, LogLevelName(level), numericThreadId);
                }
                printf("%s", buffer);
                fflush(stdout);
            }
        private:
            LogLevel m_level;
            std::chrono::steady_clock::time_point m_start;
            std::mutex m_lock;
        };

        class FileLogger : public Logger {
        public:
            FileLogger(LogLevel level, const char* file)
                : m_level(level), m_start(std::chrono::steady_clock::now())
            {
                m_handle.reset(new std::fstream(file, std::ios::out));
            }

            ~FileLogger()
            {
                if (m_handle != nullptr) m_handle->close();
            }

            virtual void Logging(const char* title, LogLevel level, const char* file, int line, const char* func, const char* format, ...)
            {
                if (level < m_level || m_handle == nullptr || !m_handle->is_open()) return;

                va_list args;
                va_start(args, format);

                char buffer[4096];
                int ret = vsnprintf(buffer, sizeof(buffer), format, args);
                va_end(args);
                if (ret < 0) return;

                auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - m_start)
                                     .count();
                long long sec = elapsedMs / 1000;
                long long ms = elapsedMs % 1000;
                int numericThreadId = GetNumericThreadId();

                char prefix[160];
                int prefixLen = snprintf(prefix, sizeof(prefix), "[+%lld.%03llds][%s][thread %d] ", sec, ms,
                                         LogLevelName(level), numericThreadId);

                std::lock_guard<std::mutex> lock(m_lock);
                if (level != LogLevel::LL_Empty && prefixLen > 0)
                {
                    m_handle->write(prefix, static_cast<std::streamsize>(prefixLen));
                }
                m_handle->write(buffer, strlen(buffer));
                m_handle->flush();
            }
        private:
            LogLevel m_level;
            std::chrono::steady_clock::time_point m_start;
            std::unique_ptr<std::fstream> m_handle;
            std::mutex m_lock;
        };
    } // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_LOGGING_H_
