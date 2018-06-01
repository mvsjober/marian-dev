#pragma once

#include <deque>
#include <functional>
#include <queue>

#include <boost/timer/timer.hpp>

#include "common/config.h"
#include "data/batch_stats.h"
#include "data/vocab.h"

namespace marian {

namespace data {

template <class DataSet>
class BatchGenerator {
public:
  typedef typename DataSet::batch_ptr BatchPtr;

  typedef typename DataSet::sample sample;
  typedef std::vector<sample> samples;

private:
  Ptr<DataSet> data_;
  Ptr<Config> options_;
  Ptr<BatchStats> stats_;

  int batchSize_{1};

  size_t visualFeatureOffset_{0};

  typename DataSet::iterator current_;

  size_t maxiBatchSize_;
  std::deque<BatchPtr> bufferedBatches_;
  BatchPtr currentBatch_;

  std::mt19937 g_;

  void fillBatches(bool shuffle = true) {
    auto cmpSrc = [](const sample& a, const sample& b) {
      return a[0].size() < b[0].size();
    };

    auto cmpTrg = [](const sample& a, const sample& b) {
      return a.back().size() < b.back().size();
    };

    auto cmpNone = [](const sample& a, const sample& b) { return &a < &b; };

    typedef std::function<bool(const sample&, const sample&)> cmp_type;
    typedef std::priority_queue<sample, samples, cmp_type> sample_queue;

    std::unique_ptr<sample_queue> maxiBatch;

    if(options_->has("maxi-batch-sort")) {
      if(options_->get<std::string>("maxi-batch-sort") == "src")
        maxiBatch.reset(new sample_queue(cmpSrc));
      else if(options_->get<std::string>("maxi-batch-sort") == "none")
        maxiBatch.reset(new sample_queue(cmpNone));
      else
        maxiBatch.reset(new sample_queue(cmpTrg));
    } else {
      maxiBatch.reset(new sample_queue(cmpNone));
    }

    int maxBatchSize = options_->get<int>("mini-batch");
    int maxSize = maxBatchSize * options_->get<int>("maxi-batch");

    size_t sets = 0;
    while(current_ != data_->end() && maxiBatch->size() < maxSize) {
      maxiBatch->push(*current_);
      sets = current_->size();
      current_++;
    }

    samples batchVector;
    int currentWords = 0;
    std::vector<size_t> lengths(sets, 0);

    while(!maxiBatch->empty()) {
      batchVector.push_back(maxiBatch->top());
      currentWords += batchVector.back()[0].size();
      maxiBatch->pop();

      // Batch size based on sentences
      bool makeBatch = batchVector.size() == maxBatchSize;

      // Batch size based on words
      if(options_->has("mini-batch-words")) {
        int mbWords = options_->get<int>("mini-batch-words");
        if(mbWords > 0)
          makeBatch = currentWords > mbWords;
      }

      if(options_->has("mini-batch-fit")) {
        // Dynamic batching
        if(stats_ && options_->get<bool>("mini-batch-fit")) {
          for(size_t i = 0; i < sets; ++i)
            if(batchVector.back()[i].size() > lengths[i])
              lengths[i] = batchVector.back()[i].size();

          maxBatchSize = stats_->getBatchSize(lengths);

          if(batchVector.size() > maxBatchSize) {
            maxiBatch->push(batchVector.back());
            batchVector.pop_back();
            makeBatch = true;
          } else {
            makeBatch = batchVector.size() == maxBatchSize;
          }
        }
      }

      if(makeBatch) {
        // std::cerr << "Creating batch" << std::endl;
        bufferedBatches_.push_back(data_->toBatch(batchVector));
        batchVector.clear();
        currentWords = 0;
        lengths.clear();
        lengths.resize(sets, 0);
      }
    }
    if(!batchVector.empty())
      bufferedBatches_.push_back(data_->toBatch(batchVector));

    if(shuffle) {
      std::shuffle(bufferedBatches_.begin(), bufferedBatches_.end(), g_);
    }
  }

public:
  BatchGenerator(Ptr<DataSet> data,
                 Ptr<Config> options,
                 Ptr<BatchStats> stats = nullptr)
      : data_(data), options_(options), stats_(stats), g_(Config::seed)
  {
    visualFeatureOffset_ = data->visualFeatureOffset();
  }

  operator bool() const { return !bufferedBatches_.empty(); }

  BatchPtr next() {
    ABORT_IF(bufferedBatches_.empty(), "No batches to fetch, run prepare()");
    currentBatch_ = bufferedBatches_.front();
    bufferedBatches_.pop_front();

    if(bufferedBatches_.empty())
      fillBatches();

    currentBatch_->setVisualFeatureOffset(visualFeatureOffset_);
    return currentBatch_;
  }

  void prepare(bool shuffle = true) {
    if(shuffle)
      data_->shuffle();
    else
      data_->reset();
    current_ = data_->begin();
    fillBatches(shuffle);
  }
};
}
}
