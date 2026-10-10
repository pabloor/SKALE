#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "analysis/Analyzer.h"
#include "analysis/AudioFile.h"

namespace {

void usage() {
    std::fprintf(stderr,
                 "Uso: skale-cli <archivo.wav|archivo.mp3> [opciones]\n"
                 "  --json            salida en JSON\n"
                 "  --model <classic|learned>  modelo de tonalidad: ensemble (por defecto: redes + modelo lineal según --cnn-weight), cnn (solo las redes), learned (solo el lineal) o classic\n"
                 "  --doubt <r>                dudosa (se dan dos tonalidades) si la 2ª tiene >= r veces la probabilidad de la 1ª (por defecto 0.5; 0 = nunca)\n"
                 "  --cnn-weight <w>           peso de las redes frente al lineal en ensemble (por defecto 1 = solo las redes)\n"
                 "  --spec <archivo.bin>       vuelca el espectro logarítmico por fotograma (float32 [n][216])\n"
                 "  --frames <archivo.bin>     vuelca la serie de cromagramas finos (float32, para redes)\n"
                 "  --features        volcado de cromagramas y uso de acordes (JSON, para experimentos)\n"
                 "  --timeline        muestra la línea de tiempo de acordes\n"
                 "  --solfege         Do Re Mi en lugar de C D E\n"
                 "  --profile <ks|temperley>   perfil de tonalidad (por defecto temperley)\n"
                 "  --ending-weight <w>        peso del acorde final en la tonalidad (0 = off, por defecto 0.5)\n"
                 "  --bass-weight <w>          peso del bajo en la tonalidad (0 = off, por defecto 1)\n"
                 "  --chord-weight <w>         peso de los acordes detectados en la tonalidad (0 = off, por defecto)\n"
                 "  --ending-margin <m>        el final solo desempata tonalidades a <m de la mejor (por defecto sin límite)\n"
                 "  --ending-seconds <s>       segundos finales que se miran (por defecto 4)\n");
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
    if (a.doubtful)
        std::printf("Tonalidad:  %s o %s  (dudosa: %.0f%% / %.0f%%%s)\n", a.keyName.c_str(), a.secondKeyName.c_str(),
                    double(a.candidates[0].confidence) * 100, double(a.candidates[1].confidence) * 100,
                    a.secondSameNotes ? ", mismas notas" : "");
    else
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
    if (a.doubtful && !a.secondSameNotes) {
        std::printf("\n  o, si es %s: ", a.secondKeyName.c_str());
        for (const auto& n : a.secondScaleNotes) std::printf("%s ", n.c_str());
    }
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
    std::printf("  \"doubtful\": %s,\n", a.doubtful ? "true" : "false");
    if (a.doubtful)
        std::printf("  \"secondKey\": {\"name\": %s, \"tonic\": %d, \"mode\": \"%s\", \"sameNotes\": %s},\n",
                    jsonString(a.secondKeyName).c_str(), a.secondKey.tonic, a.secondKey.mode == skale::Mode::Major ? "major" : "minor",
                    a.secondSameNotes ? "true" : "false");
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
    bool json = false, timeline = false, features = false;
    const char* framesOut = nullptr;
    const char* specOut = nullptr;
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
        } else if (!std::strcmp(arg, "--ending-weight") && i + 1 < argc) {
            options.endingWeight = std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--model") && i + 1 < argc) {
            const char* mname = argv[++i];
            if (!std::strcmp(mname, "ensemble")) { options.cnn = true; options.learnedModel = true; }
            else if (!std::strcmp(mname, "learned")) { options.learnedModel = true; options.cnn = false; }
            else if (!std::strcmp(mname, "classic")) { options.learnedModel = false; options.cnn = false; }
            else if (!std::strcmp(mname, "cnn")) { options.cnn = true; options.learnedModel = false; }
            else { usage(); return 2; }
        } else if (!std::strcmp(arg, "--cnn-weight") && i + 1 < argc) {
            options.cnnWeight = std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--doubt") && i + 1 < argc) {
            options.doubtRatio = std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--chroma-gamma") && i + 1 < argc) {
            options.chroma.gamma = std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--chroma-min") && i + 1 < argc) {
            options.chroma.minFreq = std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--chroma-max") && i + 1 < argc) {
            options.chroma.maxFreq = std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--peak-floor") && i + 1 < argc) {
            options.chroma.peakFloor = std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--window") && i + 1 < argc) {
            options.windowSeconds = std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--spec") && i + 1 < argc) {
            specOut = argv[++i];
            options.keepFrames = true;
        } else if (!std::strcmp(arg, "--frames") && i + 1 < argc) {
            framesOut = argv[++i];
            options.keepFrames = true;
        } else if (!std::strcmp(arg, "--features")) {
            features = true;
        } else if (!std::strcmp(arg, "--bass-weight") && i + 1 < argc) {
            options.bassWeight = std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--chord-weight") && i + 1 < argc) {
            options.chordWeight = std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--ending-margin") && i + 1 < argc) {
            options.endingMargin = std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--ending-seconds") && i + 1 < argc) {
            options.endingSeconds = std::atof(argv[++i]);
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

    if (specOut) {
        // Espectro logarítmico por fotograma: float32 [n][216].
        std::FILE* fh = std::fopen(specOut, "wb");
        if (!fh) { std::fprintf(stderr, "no se pudo escribir %s\n", specOut); return 1; }
        for (const auto& f : a.frames) std::fwrite(f.spec.data(), sizeof(float), skale::kSpecBins, fh);
        std::fclose(fh);
        if (!framesOut) return 0;
    }
    if (framesOut) {
        // Serie de cromagramas para redes neuronales: float32 [n][4][36] (cromagrama, bajo, medios, agudos).
        std::FILE* fh = std::fopen(framesOut, "wb");
        if (!fh) { std::fprintf(stderr, "no se pudo escribir %s\n", framesOut); return 1; }
        for (const auto& f : a.frames) {
            std::fwrite(f.chroma.data(), sizeof(float), 36, fh);
            std::fwrite(f.bass.data(), sizeof(float), 36, fh);
            std::fwrite(f.mid.data(), sizeof(float), 36, fh);
            std::fwrite(f.high.data(), sizeof(float), 36, fh);
        }
        std::fclose(fh);
        return 0;
    }
    if (features) {
        // Volcado para experimentar con modelos fuera de C++: cromagramas y uso de acordes.
        auto arr = [](const skale::Chroma12& c) {
            std::printf("[");
            for (int i = 0; i < 12; ++i) std::printf("%s%.6f", i ? "," : "", double(c[std::size_t(i)]));
            std::printf("]");
        };
        std::printf("{\"chroma\":"); arr(a.chroma);
        std::printf(",\"bass\":"); arr(a.bassChroma);
        std::printf(",\"ending\":"); arr(a.endingChroma);
        std::printf(",\"start\":"); arr(a.startChroma);
        std::printf(",\"duration\":%.2f,\"chords\":[", a.durationSeconds);
        for (std::size_t i = 0; i < a.chordUsage.size(); ++i) {
            const auto& u = a.chordUsage[i];
            std::printf("%s[%d,%d,%.3f]", i ? "," : "", u.chord.root, int(u.chord.type), u.seconds);
        }
        std::printf("],\"win\":[");
        for (std::size_t i = 0; i < a.windows.size(); ++i) {
            std::printf("%s[", i ? "," : "");
            arr(a.windows[i].chroma); std::printf(","); arr(a.windows[i].bass);
            std::printf("]");
        }
        std::printf("],\"seq\":[");
        bool first = true;
        for (const auto& e : a.timeline) {
            if (e.chord.none) continue;
            std::printf("%s[%d,%d,%.2f,%.2f]", first ? "" : ",", e.chord.root, int(e.chord.type), e.start, e.end);
            first = false;
        }
        std::printf("]}\n");
        return 0;
    }
    if (json) printJson(a, timeline);
    else printText(a, timeline);
    return 0;
}
