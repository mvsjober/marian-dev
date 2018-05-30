#pragma once

#include <vector>

#include "common/definitions.h"

namespace marian {
namespace data {

class Batch {
public:
  virtual size_t size() const = 0;
  virtual size_t words() const { return 0; };
  virtual void debug(){};

  virtual std::vector<Ptr<Batch>> split(size_t n) = 0;

  const std::vector<size_t>& getSentenceIds() const { return sentenceIds_; }
  void setSentenceIds(const std::vector<size_t>& ids) { sentenceIds_ = ids; }

  size_t visualFeatureOffset() const { return visualFeatureOffset_; }
  void setVisualFeatureOffset(size_t o) { visualFeatureOffset_ = o; }

protected:
  size_t visualFeatureOffset_ {0};
  std::vector<size_t> sentenceIds_;
};
}
}
