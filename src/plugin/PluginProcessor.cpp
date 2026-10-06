#include "PluginProcessor.h"

#include "PluginEditor.h"

#include <juce_audio_formats/juce_audio_formats.h>

class FileAnalysisJob : public juce::ThreadPoolJob {
public:
    FileAnalysisJob(SkaleProcessor& p, juce::File f, int id)
        : juce::ThreadPoolJob("Skale file analysis"), proc_(p), file_(std::move(f)), id_(id) {}

    JobStatus runJob() override {
        FileAnalysisState st;
        st.fileName = file_.getFileName();
        juce::AudioFormatManager fm;
        fm.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader(fm.createReaderFor(file_));
        if (!reader || reader->lengthInSamples <= 0) {
            st.status = FileAnalysisState::Status::Error;
            st.error = "No se pudo leer el archivo (formatos: wav, aiff, flac, ogg, mp3).";
            return finish(st);
        }

        // A mono en trozos, para no necesitar el doble de memoria.
        std::vector<float> mono;
        mono.reserve(std::size_t(reader->lengthInSamples));
        juce::AudioBuffer<float> buf(int(std::max<unsigned>(1u, reader->numChannels)), 65536);
        for (juce::int64 pos = 0; pos < reader->lengthInSamples; pos += 65536) {
            if (shouldExit() || proc_.fileJobId_.load() != id_) return jobHasFinished;
            const int n = int(std::min<juce::int64>(65536, reader->lengthInSamples - pos));
            reader->read(&buf, 0, n, pos, true, true);
            const int ch = buf.getNumChannels();
            for (int i = 0; i < n; ++i) {
                float sum = 0.f;
                for (int c = 0; c < ch; ++c) sum += buf.getReadPointer(c)[i];
                mono.push_back(sum / float(ch));
            }
        }

        st.result = skale::analyze(mono.data(), mono.size(), reader->sampleRate);
        if (!st.result.valid) {
            st.status = FileAnalysisState::Status::Error;
            st.error = "El archivo no tiene audio utilizable (silencio).";
        } else {
            st.status = FileAnalysisState::Status::Done;
        }
        return finish(st);
    }

private:
    JobStatus finish(FileAnalysisState& st) {
        if (proc_.fileJobId_.load() != id_) return jobHasFinished;   // ya hay otro archivo en curso
        std::lock_guard<std::mutex> lock(proc_.fileMutex_);
        proc_.fileState_ = std::move(st);
        return jobHasFinished;
    }

    SkaleProcessor& proc_;
    juce::File file_;
    int id_;
};

SkaleProcessor::SkaleProcessor()
    : juce::AudioProcessor(BusesProperties()
                               .withInput("Input", juce::AudioChannelSet::stereo(), true)
                               .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      juce::Thread("Skale analysis"),
      fifoData_(kFifoSize, 0.f) {
    startThread(juce::Thread::Priority::low);
}

SkaleProcessor::~SkaleProcessor() {
    fileJobId_.fetch_add(1);
    filePool_.removeAllJobs(true, 5000);
    stopThread(2000);
}

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

void SkaleProcessor::analyzeFile(const juce::File& file) {
    const int id = fileJobId_.fetch_add(1) + 1;
    {
        std::lock_guard<std::mutex> lock(fileMutex_);
        fileState_ = {};
        fileState_.status = FileAnalysisState::Status::Working;
        fileState_.fileName = file.getFileName();
    }
    filePool_.addJob(new FileAnalysisJob(*this, file, id), true);
}

FileAnalysisState SkaleProcessor::fileState() const {
    std::lock_guard<std::mutex> lock(fileMutex_);
    return fileState_;
}

void SkaleProcessor::clearFileAnalysis() {
    fileJobId_.fetch_add(1);
    std::lock_guard<std::mutex> lock(fileMutex_);
    fileState_ = {};
}

juce::AudioProcessorEditor* SkaleProcessor::createEditor() { return new SkaleEditor(*this); }

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new SkaleProcessor(); }
