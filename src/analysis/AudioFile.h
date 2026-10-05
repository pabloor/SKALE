#pragma once

#include <string>
#include <vector>

namespace skale {

struct AudioData {
    std::vector<float> mono;  // mezcla a mono (media de canales)
    double sampleRate = 0;
    int channels = 0;
};

// Carga un wav o mp3 completo. Devuelve false y rellena `error` si falla.
bool loadAudioFile(const std::string& path, AudioData& out, std::string& error);

}  // namespace skale
