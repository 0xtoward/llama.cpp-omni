#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <dirent.h>
#endif

namespace omni::wav_chunks {

inline bool parse_filename(const std::string & name, uint64_t & index) {
    static const std::string prefix = "wav_";
    static const std::string suffix = ".wav";
    if (name.size() <= prefix.size() + suffix.size() ||
        name.compare(0, prefix.size(), prefix) != 0 ||
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return false;
    }
    const std::string digits =
        name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
    if (digits.empty() ||
        !std::all_of(digits.begin(), digits.end(), [](unsigned char value) {
            return std::isdigit(value) != 0;
        })) {
        return false;
    }
    try {
        index = std::stoull(digits);
    } catch (...) {
        return false;
    }
    return true;
}

inline bool is_nonempty_file(const std::string & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    return stream.good() && stream.tellg() > 0;
}

inline std::vector<std::string> list(const std::string & directory) {
    std::vector<std::pair<uint64_t, std::string>> indexed;
#ifdef _WIN32
    struct _finddata_t entry;
    const std::string pattern = directory + "\\wav_*.wav";
    intptr_t handle = _findfirst(pattern.c_str(), &entry);
    if (handle != -1) {
        do {
            uint64_t index = 0;
            const std::string name(entry.name);
            const std::string path = directory + "/" + name;
            if (parse_filename(name, index) && is_nonempty_file(path)) {
                indexed.emplace_back(index, path);
            }
        } while (_findnext(handle, &entry) == 0);
        _findclose(handle);
    }
#else
    DIR * handle = opendir(directory.c_str());
    if (handle != nullptr) {
        while (dirent * entry = readdir(handle)) {
            uint64_t index = 0;
            const std::string name(entry->d_name);
            const std::string path = directory + "/" + name;
            if (parse_filename(name, index) && is_nonempty_file(path)) {
                indexed.emplace_back(index, path);
            }
        }
        closedir(handle);
    }
#endif
    std::sort(indexed.begin(), indexed.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        return lhs.second < rhs.second;
    });
    std::vector<std::string> result;
    result.reserve(indexed.size());
    for (auto & item : indexed) {
        result.push_back(std::move(item.second));
    }
    return result;
}

} // namespace omni::wav_chunks
