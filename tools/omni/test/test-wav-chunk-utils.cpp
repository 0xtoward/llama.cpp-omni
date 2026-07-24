#include "wav-chunk-utils.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

static void write_fixture(const fs::path & path, const std::string & content) {
    std::ofstream stream(path, std::ios::binary);
    stream << content;
}

int main() {
    uint64_t index = 0;
    assert(omni::wav_chunks::parse_filename("wav_5000.wav", index));
    assert(index == 5000);
    assert(!omni::wav_chunks::parse_filename("tts_output_chunk_0.wav", index));
    assert(!omni::wav_chunks::parse_filename("wav_not-a-number.wav", index));

    const fs::path root = fs::temp_directory_path() / "omni-wav-chunk-utils-test";
    fs::remove_all(root);
    fs::create_directories(root);
    write_fixture(root / "wav_10.wav", "ten");
    write_fixture(root / "wav_2.wav", "two");
    write_fixture(root / "wav_1000.wav", "thousand");
    write_fixture(root / "wav_3.wav", "");
    write_fixture(root / "tts_output_chunk_0.wav", "legacy");

    const auto chunks = omni::wav_chunks::list(root.string());
    assert(chunks.size() == 3);
    assert(fs::path(chunks[0]).filename() == "wav_2.wav");
    assert(fs::path(chunks[1]).filename() == "wav_10.wav");
    assert(fs::path(chunks[2]).filename() == "wav_1000.wav");
    fs::remove_all(root);
    return 0;
}
