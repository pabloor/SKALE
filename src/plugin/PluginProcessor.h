#pragma once

#include <atomic>
#include <memory>
#include <mutex>

#include <juce_audio_processors/juce_audio_processors.h>

#include "analysis/KeyTracker.h"

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
};
