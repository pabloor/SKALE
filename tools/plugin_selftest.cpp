// Prueba del plugin sin anfitrión. Guarda una captura de la interfaz con el resultado.
// Uso: SkalePluginSelfTest --file audio captura.png      (como al arrastrar el archivo)
//      SkalePluginSelfTest audio captura.png [segundos]  (como «Analyze song»: pasa el audio por processBlock en
//                                                          bloques de 512, termina la escucha y espera el análisis)
#include <string>

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include "../src/plugin/PluginEditor.h"
#include "../src/plugin/PluginProcessor.h"

static int finish(SkaleProcessor& proc, const char* png) {
    for (int i = 0; i < 1200 && proc.state().status == SongState::Status::Working; ++i) juce::Thread::sleep(100);
    const auto st = proc.state();
    if (st.status != SongState::Status::Done) {
        std::printf("estado=%d error=%s\n", int(st.status), st.error.toRawUTF8());
        return 1;
    }
    const auto& r = st.result;
    std::printf("origen=%s tonalidad=%s confianza=%.0f%% duracion=%.1fs escala=", st.source.toRawUTF8(), r.keyName.c_str(),
                r.candidates.empty() ? 0.0 : r.candidates[0].confidence * 100.0, r.durationSeconds);
    for (auto& n : r.scaleNotes) std::printf("%s ", n.c_str());
    std::printf(" 2a=%s\n", r.secondKeyName.c_str());

    std::unique_ptr<juce::AudioProcessorEditor> ed(proc.createEditor());
    static_cast<SkaleEditor*>(ed.get())->refresh();
    auto img = ed->createComponentSnapshot(ed->getLocalBounds());
    juce::File out(png);
    out.deleteFile();
    juce::FileOutputStream os(out);
    juce::PNGImageFormat().writeImageToStream(img, os);
    return 0;
}

int main(int argc, char** argv) {
    juce::ScopedJuceInitialiser_GUI gui;
    if (argc >= 4 && std::string(argv[1]) == "--file") {
        SkaleProcessor proc;
        proc.setPlayConfigDetails(2, 2, 44100.0, 512);
        proc.analyzeFile(juce::File(argv[2]));
        return finish(proc, argv[3]);
    }
    if (argc < 3) { std::printf("uso: %s [--file] audio captura.png [segundos]\n", argv[0]); return 2; }
    const double maxSeconds = argc > 3 ? std::atof(argv[3]) : 600.0;

    juce::AudioFormatManager fm;
    fm.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader(fm.createReaderFor(juce::File(argv[1])));
    if (!reader) { std::printf("no se pudo leer %s\n", argv[1]); return 2; }

    SkaleProcessor proc;
    proc.setPlayConfigDetails(2, 2, reader->sampleRate, 512);
    proc.prepareToPlay(reader->sampleRate, 512);
    proc.startListening();   // sin cabezal de reproducción (como la aplicación independiente): escucha todo

    const juce::int64 total = std::min<juce::int64>(reader->lengthInSamples, juce::int64(maxSeconds * reader->sampleRate));
    juce::AudioBuffer<float> buf(2, 512);
    juce::MidiBuffer midi;
    for (juce::int64 pos = 0; pos < total; pos += 512) {
        const int n = int(std::min<juce::int64>(512, total - pos));
        buf.setSize(2, 512, false, false, true);
        buf.clear();
        reader->read(&buf, 0, n, pos, true, true);
        if (reader->numChannels == 1) buf.copyFrom(1, 0, buf, 0, 0, n);
        buf.setSize(2, n, true, false, true);
        proc.processBlock(buf, midi);
    }
    proc.stopListening();
    return finish(proc, argv[2]);
}
