#include "analysis/AudioFile.h"

#include <algorithm>
#include <cctype>
#include <cstdint>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#define DR_MP3_IMPLEMENTATION
#include "dr_mp3.h"

namespace skale {

namespace {

std::string extensionOf(const std::string& path) {
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return {};
    std::string ext = path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return ext;
}

void downmix(const float* interleaved, std::uint64_t frames, unsigned channels, std::vector<float>& mono) {
    mono.resize(std::size_t(frames));
    for (std::uint64_t i = 0; i < frames; ++i) {
        float sum = 0;
        for (unsigned c = 0; c < channels; ++c) sum += interleaved[i * channels + c];
        mono[std::size_t(i)] = sum / float(channels);
    }
}

}  // namespace

bool loadAudioFile(const std::string& path, AudioData& out, std::string& error) {
    const std::string ext = extensionOf(path);

    if (ext == "wav") {
        unsigned channels = 0, rate = 0;
        drwav_uint64 frames = 0;
        float* data = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &channels, &rate, &frames, nullptr);
        if (!data) { error = "no se pudo leer el wav: " + path; return false; }
        downmix(data, frames, channels, out.mono);
        out.sampleRate = rate;
        out.channels = int(channels);
        drwav_free(data, nullptr);
        return true;
    }

    if (ext == "mp3") {
        drmp3_config cfg{};
        drmp3_uint64 frames = 0;
        float* data = drmp3_open_file_and_read_pcm_frames_f32(path.c_str(), &cfg, &frames, nullptr);
        if (!data) { error = "no se pudo leer el mp3: " + path; return false; }
        downmix(data, frames, cfg.channels, out.mono);
        out.sampleRate = cfg.sampleRate;
        out.channels = int(cfg.channels);
        drmp3_free(data, nullptr);
        return true;
    }

    error = "formato no soportado (solo .wav y .mp3); convierte antes con ffmpeg";
    return false;
}

}  // namespace skale
