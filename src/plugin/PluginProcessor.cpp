#include "PluginProcessor.h"

#include "PluginEditor.h"

SkaleProcessor::SkaleProcessor()
    : juce::AudioProcessor(BusesProperties()
                               .withInput("Input", juce::AudioChannelSet::stereo(), true)
                               .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      juce::Thread("Skale analysis"),
      fifoData_(kFifoSize, 0.f) {
    startThread(juce::Thread::Priority::low);
}

SkaleProcessor::~SkaleProcessor() { stopThread(2000); }

void SkaleProcessor::prepareToPlay(double sampleRate, int) {
    sampleRate_.store(sampleRate);
    rateChanged_.store(true);
    fifo_.reset();
}

bool SkaleProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto& in = layouts.getMainInputChannelSet();
    return in == layouts.getMainOutputChannelSet() && !in.isDisabled();
}

void SkaleProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) {
    juce::ScopedNoDenormals noDenormals;
    const int n = buffer.getNumSamples();
    const int channels = buffer.getNumChannels();
    if (channels == 0 || n == 0) return;

    int start1, size1, start2, size2;
    fifo_.prepareToWrite(n, start1, size1, start2, size2);   // si no cabe, se descarta lo que sobre
    const float gain = 1.0f / float(channels);
    auto write = [&](int dst, int count, int srcOffset) {
        for (int i = 0; i < count; ++i) {
            float s = 0.f;
            for (int c = 0; c < channels; ++c) s += buffer.getReadPointer(c)[srcOffset + i];
            fifoData_[std::size_t(dst + i)] = s * gain;
        }
    };
    write(start1, size1, 0);
    if (size2 > 0) write(start2, size2, size1);
    fifo_.finishedWrite(size1 + size2);
    // El audio pasa sin modificar.
}

void SkaleProcessor::run() {
    std::unique_ptr<skale::KeyTracker> tracker;
    std::vector<float> block(4096);
    double lastPublish = 0;

    while (!threadShouldExit()) {
        if (rateChanged_.exchange(false) || !tracker) {
            tracker = std::make_unique<skale::KeyTracker>(sampleRate_.load());
        }
        if (resetRequested_.exchange(false)) {
            tracker->reset();
            std::lock_guard<std::mutex> lock(snapMutex_);
            snapshot_ = {};
        }

        const int ready = fifo_.getNumReady();
        if (ready > 0) {
            const int count = std::min<int>(ready, int(block.size()));
            int start1, size1, start2, size2;
            fifo_.prepareToRead(count, start1, size1, start2, size2);
            std::copy_n(fifoData_.begin() + start1, size1, block.begin());
            if (size2 > 0) std::copy_n(fifoData_.begin() + start2, size2, block.begin() + size1);
            fifo_.finishedRead(size1 + size2);
            tracker->process(block.data(), std::size_t(size1 + size2));
        } else {
            wait(10);
        }

        const double now = juce::Time::getMillisecondCounterHiRes() * 0.001;
        if (now - lastPublish > 0.2) {
            lastPublish = now;
            auto snap = tracker->snapshot();
            std::lock_guard<std::mutex> lock(snapMutex_);
            snapshot_ = std::move(snap);
        }
    }
}

skale::TrackerSnapshot SkaleProcessor::latest() const {
    std::lock_guard<std::mutex> lock(snapMutex_);
    return snapshot_;
}

juce::AudioProcessorEditor* SkaleProcessor::createEditor() { return new SkaleEditor(*this); }

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new SkaleProcessor(); }
