#pragma once

#include <atomic>
#include <memory>
#include <mutex>

#include <juce_audio_processors/juce_audio_processors.h>

#include "analysis/Analyzer.h"

// Escala de una canción entera, para ajustar Auto-Tune, Melodyne, etc. Tres formas de darle la canción:
//  - ARA (Logic, Studio One, Cubase, Reaper...): lee el audio completo de las regiones de la pista, sin reproducir.
//  - Sin ARA: «Analizar canción» captura el audio mientras suena y, al parar la reproducción, analiza todo.
//  - Arrastrar o abrir un archivo de audio.
// El resultado es uno solo para toda la canción (el mismo análisis que skale-cli) y no cambia mientras suena.
struct SongState {
    enum class Status { Idle, Listening, Working, Done, Error };
    Status status = Status::Idle;
    juce::String source;            // de dónde sale: archivo, región(es) de la pista, audio escuchado
    juce::String error;
    double listenedSeconds = 0;     // mientras se escucha
    skale::SongAnalysis result;
};

class SkaleProcessor : public juce::AudioProcessor,
#if JucePlugin_Enable_ARA
                       public juce::AudioProcessorARAExtension,
#endif
                       private juce::Timer {
public:
    SkaleProcessor();
    ~SkaleProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Skale"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override;
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}
    void getStateInformation(juce::MemoryBlock&) override {}
    void setStateInformation(const void*, int) override {}

    SongState state() const;
    bool usesARA() const;

    void analyzeFile(const juce::File& file);
    // Sin ARA: empieza a escuchar (hasta kMaxListenSeconds); termina sola al parar la reproducción o con stopListening().
    void startListening();
    void stopListening();
    void clear();

    // Análisis de audio mono ya leído (en el hilo de análisis). Lo usan los tres caminos.
    void analyzeSamples(std::vector<float> mono, double sampleRate, const juce::String& source);

    static constexpr double kMaxListenSeconds = 600;

private:
    void timerCallback() override;
#if JucePlugin_Enable_ARA
    void analyzeRegionsIfChanged();
    juce::String regionsSignature_;
#endif

    // Captura (sin ARA): el hilo de audio solo escribe en un vector ya reservado.
    std::vector<float> capture_;
    juce::SpinLock captureLock_;               // el hilo de audio solo lo intenta; si no lo consigue, no escribe
    std::atomic<std::size_t> captured_{0};
    std::atomic<bool> capturing_{false};
    std::atomic<bool> heardPlayback_{false};   // el anfitrión ha reproducido desde que se pulsó «Analizar»
    std::atomic<bool> playbackStopped_{false};
    std::atomic<double> sampleRate_{44100.0};

    mutable std::mutex stateMutex_;
    SongState state_;
    std::atomic<int> jobId_{0};                // un análisis nuevo invalida el anterior
    juce::ThreadPool pool_{1};
    friend class SongAnalysisJob;
};

#if JucePlugin_Enable_ARA
const ARA::ARAFactory* JUCE_CALLTYPE createARAFactory();
#endif
