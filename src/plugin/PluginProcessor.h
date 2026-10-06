#pragma once

#include <atomic>
#include <memory>
#include <mutex>

#include <juce_audio_processors/juce_audio_processors.h>

#include "analysis/Analyzer.h"
#include "analysis/KeyTracker.h"

struct FileAnalysisState {
    enum class Status { Idle, Working, Done, Error };
    Status status = Status::Idle;
    juce::String fileName;
    juce::String error;
    skale::SongAnalysis result;
};

// El hilo de audio solo copia muestras (mono) a un buffer circular sin bloqueos.
// Un hilo de análisis lee ese buffer, hace las FFT con KeyTracker y publica una
// instantánea que el editor consulta.
class SkaleProcessor : public juce::AudioProcessor, private juce::Thread {
public:
    SkaleProcessor();
    ~SkaleProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Skale"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}
    void getStateInformation(juce::MemoryBlock&) override {}
    void setStateInformation(const void*, int) override {}

    skale::TrackerSnapshot latest() const;
    void resetAnalysis() { resetRequested_.store(true); }

    // Análisis completo de un archivo de audio (en un hilo aparte).
    void analyzeFile(const juce::File& file);
    FileAnalysisState fileState() const;
    void clearFileAnalysis();

private:
    void run() override;

    static constexpr int kFifoSize = 1 << 18;   // ~6 s a 44,1 kHz; el hilo de análisis va muy por delante
    juce::AbstractFifo fifo_{kFifoSize};
    std::vector<float> fifoData_;

    std::atomic<double> sampleRate_{44100.0};
    std::atomic<bool> rateChanged_{true};
    std::atomic<bool> resetRequested_{false};

    mutable std::mutex snapMutex_;
    skale::TrackerSnapshot snapshot_;

    mutable std::mutex fileMutex_;
    FileAnalysisState fileState_;
    std::atomic<int> fileJobId_{0};     // un archivo nuevo invalida el análisis anterior
    juce::ThreadPool filePool_{1};
    friend class FileAnalysisJob;
};
