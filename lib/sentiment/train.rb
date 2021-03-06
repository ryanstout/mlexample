# jruby  -J-server -J-Xmx4000m -S train.rb
# Build status info cache model averages after


# http://weka.wikispaces.com/Use+WEKA+in+your+Java+code

# Load java libs
require 'java'
require File.expand_path(File.join(File.dirname(__FILE__), '..', 'weka-3-7-5', 'weka.jar'))
require File.expand_path(File.join(File.dirname(__FILE__), '/../weka-3-7-5/packages/packages/LibSVM/LibSVM.jar'))
require File.expand_path(File.join(File.dirname(__FILE__), '/../weka-3-7-5/packages/packages/LibSVM/lib/libsvm.jar'))
require File.expand_path(File.join(File.dirname(__FILE__), '/../weka-3-7-5/packages/packages/LibLINEAR/LibLINEAR.jar'))
require File.expand_path(File.join(File.dirname(__FILE__), '/../weka-3-7-5/packages/packages/LibSVM/lib/libsvm.jar'))
require File.expand_path(File.join(File.dirname(__FILE__), '/../weka-3-7-5/packages/packages/LibLINEAR/lib/liblinear-1.8.jar'))

# Include classes
include_class "weka.core.Instances"
include_class "weka.core.Instance"
include_class "weka.classifiers.Evaluation"
include_class "weka.classifiers.meta.FilteredClassifier"
include_class "java.io.FileReader"
include_class "weka.core.Utils"


arff_file = File.dirname(__FILE__) + "/../../data/sentiment.arff"

# Setup options (can copy in from weka)

# sets 50k dictionary, bi-grams
options = "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.StringToWordVector -R first-last -P wv- -W 50000 -prune-rate -1.0 -T -N 0 -L -stemmer weka.core.stemmers.NullStemmer -M 1 -tokenizer \\\"weka.core.tokenizers.NGramTokenizer -delimiters \\\\\\\" \\\\\\\\\\r\\\\\\\\\\n\\\\\\\\\\t.,;:\\\\\\\\\\'\\\\\\\\\\\\\\\"()?!\\\\\\\" -max 2 -min 1\\\"\" -W weka.classifiers.functions.LibLINEAR -- -S 0 -C 1.0 -E 0.01 -B 1.0 -P"

# sets 1k dictionary, uni-grams
# options = "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.StringToWordVector -R first-last -P wv- -W 1000 -prune-rate -1.0 -T -N 0 -L -stemmer weka.core.stemmers.NullStemmer -M 1 \" -W weka.classifiers.functions.LibLINEAR -- -S 0 -C 1.0 -E 0.01 -B 1.0 -P"

# Read arff
arff = FileReader.new(arff_file)
data = Instances.new(arff)
data.setClassIndex data.num_attributes - 1 if data.class_index == -1

# Setup five fold cross validation
train = data.trainCV(5, 1)
test = data.testCV(5, 1)

# Train a model as a FilteredClassifier
puts "\n----------------------------------\nBuilding sentiment model"
classifier = FilteredClassifier.new
opts = Utils.splitOptions(options)
classifier.set_options(opts)
classifier.build_classifier(data)

# Build evaluation Models
evaluation = Evaluation.new(train)
evaluation.cross_validate_model(classifier, train, 10, java.util.Random.new(1))
puts evaluation.to_summary_string("\nResults\n======\n", false)
puts "Percent Correct: " + evaluation.pct_correct.to_f.inspect

# Write out model
model_path = File.expand_path(File.join(File.dirname(__FILE__), "/../../models/sentiment.model"))
puts "Writing sentiment classifier to #{model_path}"
oos = java.io.ObjectOutputStream.new(java.io.FileOutputStream.new(model_path))
oos.writeObject(classifier)
oos.flush()
oos.close()
