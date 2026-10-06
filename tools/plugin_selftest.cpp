// Prueba del plugin sin anfitrión: pasa un wav/mp3 por SkaleProcessor::processBlock
// en bloques de 512 (como un anfitrión), espera al hilo de análisis, imprime la
// tonalidad y guarda una captura de la interfaz.
// Uso: SkalePluginSelfTest audio captura.png [segundos_maximos]
//      SkalePluginSelfTest --file audio captura.png   (análisis de archivo, como al arrastrarlo)
#include <string>

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include "../src/plugin/PluginEditor.h"
#include "../src/plugin/PluginProcessor.h"

int main(int argc, char** argv) {
    juce::ScopedJuceInitialiser_GUI gui;
    if (argc >= 4 && std::string(argv[1]) == "--file") {
        SkaleProcessor proc;
        proc.setPlayConfigDetails(2, 2, 44100.0, 512);
        proc.analyzeFile(juce::File(argv[2]));
        for (int i = 0; i < 600 && proc.fileState().status == FileAnalysisState::Status::Working; ++i) juce::Thread::sleep(100);
        const auto st = proc.fileState();
        if (st.status != FileAnalysisState::Status::Done) {
            std::printf("estado=%d error=%s\n", int(st.status), st.error.toRawUTF8());
            return 1;
        }
        std::printf("archivo=%s tonalidad=%s duracion=%.1fs acordes=%zu\n", st.fileName.toRawUTF8(),
                    st.result.keyName.c_str(), st.result.durationSeconds, st.result.chordUsage.size());
        std::unique_ptr<juce::AudioProcessorEditor> ed(proc.createEditor());
        static_cast<SkaleEditor*>(ed.get())->refresh();
        auto img = ed->createComponentSnapshot(ed->getLocalBounds());
        juce::PNGImageFormat png;
        juce::File out(argv[3]);
        out.deleteFile();
        juce::FileOutputStream os(out);
        png.writeImageToStream(img, os);
        ed.reset();
        return 0;
    }
    if (argc < 3) { std::printf("uso: %s audio captura.png [segundos]\n", argv[0]); return 2; }
    const double maxSeconds = argc > 3 ? std::atof(argv[3]) : 60.0;

    juce::AudioFormatManager fm;
    fm.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader(fm.createReaderFor(juce::File(argv[1])));
    if (!reader) { std::printf("no se pudo leer %s\n", argv[1]); return 2; }

    SkaleProcessor proc;
    proc.setPlayConfigDetails(2, 2, reader->sampleRate, 512);
    proc.prepareToPlay(reader->sampleRate, 512);

    const juce::int64 total = std::min<juce::int64>(reader->lengthInSamples, juce::int64(maxSeconds * reader->sampleRate));
    juce::AudioBuffer<float> buf(2, 512);
    juce::MidiBuffer midi;
    for (juce::int64 pos = 0; pos < total; pos += 512) {
        const int n = int(std::min<juce::int64>(512, total - pos));
        buf.clear();
        reader->read(&buf, 0, n, pos, true, true);
        if (buf.getNumChannels() > 1 && reader->numChannels == 1) buf.copyFrom(1, 0, buf, 0, 0, n);
        buf.setSize(2, n, true, false, true);
        proc.processBlock(buf, midi);
        // ritmo ~4x tiempo real (un anfitrión real va a 1x)
        juce::Thread::sleep(std::max(1, int(1000.0 * n / reader->sampleRate / 4.0)));
    }
    juce::Thread::sleep(1500);   // deja que el hilo de análisis termine y publique

    const auto s = proc.latest();
    std::printf("valid=%d escuchado=%.1fs tonalidad=%s confianza=%.0f%% acorde=%s escala=", s.valid ? 1 : 0,
                s.listenedSeconds, s.keyName.c_str(), s.confidence * 100.f, s.chordName.c_str());
    for (auto& n : s.scaleNotes) std::printf("%s ", n.c_str());
    std::printf("\n");

    std::unique_ptr<juce::AudioProcessorEditor> ed(proc.createEditor());
    static_cast<SkaleEditor*>(ed.get())->refresh();
    auto img = ed->createComponentSnapshot(ed->getLocalBounds());
    juce::PNGImageFormat png;
    juce::File out(argv[2]);
    out.deleteFile();
    juce::FileOutputStream os(out);
    png.writeImageToStream(img, os);
    ed.reset();
    return s.valid ? 0 : 1;
}
