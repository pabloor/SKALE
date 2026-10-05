#include <cstdio>
#include <cstring>
#include <string>

#include "analysis/Analyzer.h"
#include "analysis/AudioFile.h"

namespace {

void usage() {
    std::fprintf(stderr,
                 "Uso: skale-cli <archivo.wav|archivo.mp3> [opciones]\n"
                 "  --json            salida en JSON\n"
                 "  --timeline        muestra la línea de tiempo de acordes\n"
                 "  --solfege         Do Re Mi en lugar de C D E\n"
                 "  --profile <ks|temperley>   perfil de tonalidad (por defecto temperley)\n");
}

std::string clock(double t) {
    const int m = int(t) / 60;
    const double s = t - 60.0 * m;
    char buf[32];
    std::snprintf(buf, sizeof buf, "%d:%04.1f", m, s);
    return buf;
}

std::string jsonString(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (static_cast<unsigned char>(c) < 0x20) out += ' ';
        else out += c;
    }
    return out + "\"";
}

void printText(const skale::SongAnalysis& a, bool timeline) {
    std::printf("Tonalidad:  %s  (confianza %.0f%%)\n", a.keyName.c_str(), double(a.candidates[0].confidence) * 100);
    if (a.candidates.size() > 1) {
        std::printf("Alternativas:");
        for (std::size_t i = 1; i < a.candidates.size(); ++i) {
            std::printf("  %s (%.0f%%)", skale::keyName(a.candidates[i].key, false).c_str(),
                        double(a.candidates[i].confidence) * 100);
        }
        std::printf("\n");
    }
    std::printf("Afinación:  %+.0f cents respecto a A=440\n", a.tuningCents);

    std::printf("Escala:     ");
    for (const auto& n : a.scaleNotes) std::printf("%s ", n.c_str());
    std::printf("\n\nAcordes diatónicos:\n");
    for (const auto& c : a.diatonic) {
        std::printf("  %-5s %-8s %s\n", c.roman.c_str(), c.triad.c_str(), c.seventh.c_str());
    }

    std::printf("\nAcordes más usados:\n");
    int shown = 0;
    for (const auto& u : a.chordUsage) {
        if (shown++ >= 10) break;
        std::printf("  %-8s %5.1f%%  %s\n", u.name.c_str(), u.fraction * 100, u.diatonic ? "" : "(fuera de la tonalidad)");
    }

    if (timeline) {
        std::printf("\nLínea de tiempo:\n");
        for (const auto& e : a.timeline) {
            std::printf("  %s - %s  %s\n", clock(e.start).c_str(), clock(e.end).c_str(), e.name.c_str());
        }
    }
}

void printJson(const skale::SongAnalysis& a, bool timeline) {
    std::printf("{\n  \"duration\": %.2f,\n  \"tuningCents\": %.0f,\n", a.durationSeconds, a.tuningCents);
    std::printf("  \"key\": {\"name\": %s, \"tonic\": %d, \"mode\": \"%s\", \"confidence\": %.4f},\n",
                jsonString(a.keyName).c_str(), a.key.tonic, a.key.mode == skale::Mode::Major ? "major" : "minor",
                double(a.candidates[0].confidence));
    std::printf("  \"alternatives\": [");
    for (std::size_t i = 1; i < a.candidates.size(); ++i) {
        std::printf("%s{\"name\": %s, \"confidence\": %.4f}", i > 1 ? ", " : "",
                    jsonString(skale::keyName(a.candidates[i].key, false)).c_str(), double(a.candidates[i].confidence));
    }
    std::printf("],\n  \"scale\": [");
    for (std::size_t i = 0; i < a.scaleNotes.size(); ++i) {
        std::printf("%s%s", i ? ", " : "", jsonString(a.scaleNotes[i]).c_str());
    }
    std::printf("],\n  \"diatonicChords\": [");
    for (std::size_t i = 0; i < a.diatonic.size(); ++i) {
        const auto& c = a.diatonic[i];
        std::printf("%s{\"roman\": %s, \"triad\": %s, \"seventh\": %s}", i ? ", " : "", jsonString(c.roman).c_str(),
                    jsonString(c.triad).c_str(), jsonString(c.seventh).c_str());
    }
    std::printf("],\n  \"chordUsage\": [");
    for (std::size_t i = 0; i < a.chordUsage.size(); ++i) {
        const auto& u = a.chordUsage[i];
        std::printf("%s{\"chord\": %s, \"seconds\": %.2f, \"fraction\": %.4f, \"diatonic\": %s}", i ? ", " : "",
                    jsonString(u.name).c_str(), u.seconds, u.fraction, u.diatonic ? "true" : "false");
    }
    std::printf("]");
    if (timeline) {
        std::printf(",\n  \"timeline\": [");
        for (std::size_t i = 0; i < a.timeline.size(); ++i) {
            const auto& e = a.timeline[i];
            std::printf("%s{\"start\": %.2f, \"end\": %.2f, \"chord\": %s}", i ? ", " : "", e.start, e.end,
                        jsonString(e.name).c_str());
        }
        std::printf("]");
    }
    std::printf("\n}\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string path;
    bool json = false, timeline = false;
    skale::AnalysisOptions options;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (!std::strcmp(arg, "--json")) json = true;
        else if (!std::strcmp(arg, "--timeline")) timeline = true;
        else if (!std::strcmp(arg, "--solfege")) options.solfege = true;
        else if (!std::strcmp(arg, "--profile") && i + 1 < argc) {
            const char* p = argv[++i];
            if (!std::strcmp(p, "ks")) options.profile = skale::KeyProfile::KrumhanslSchmuckler;
            else if (!std::strcmp(p, "temperley")) options.profile = skale::KeyProfile::Temperley;
            else { usage(); return 2; }
        } else if (arg[0] == '-') { usage(); return 2; }
        else path = arg;
    }
    if (path.empty()) { usage(); return 2; }

    skale::AudioData audio;
    std::string error;
    if (!skale::loadAudioFile(path, audio, error)) {
        std::fprintf(stderr, "Error: %s\n", error.c_str());
        return 1;
    }

    const skale::SongAnalysis a = skale::analyze(audio.mono.data(), audio.mono.size(), audio.sampleRate, options);
    if (!a.valid) {
        std::fprintf(stderr, "No se ha podido determinar la tonalidad (¿silencio o audio demasiado corto?)\n");
        return 1;
    }

    if (json) printJson(a, timeline);
    else printText(a, timeline);
    return 0;
}
