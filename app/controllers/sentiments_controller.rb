require 'java'

# Set the weka home
ENV['WEKA_HOME'] = File.join(Rails.root, '/lib/weka-3-7-5/packages')
require Rails.root + 'lib/weka-3-7-5/packages/packages/LibSVM/LibSVM.jar'
require Rails.root + 'lib/weka-3-7-5/packages/packages/LibSVM/lib/libsvm.jar'
require Rails.root + 'lib/weka-3-7-5/packages/packages/LibLINEAR/LibLINEAR.jar'
require Rails.root + 'lib/weka-3-7-5/packages/packages/LibSVM/lib/libsvm.jar'
require Rails.root + 'lib/weka-3-7-5/packages/packages/LibLINEAR/lib/liblinear-1.8.jar'
require Rails.root + 'lib/weka-3-7-5/weka.jar'

include_class "weka.core.Instances"
include_class "weka.core.Instance"
include_class "java.io.FileReader"
include_class "weka.core.Utils"
include_class "weka.core.SerializationHelper"
include_class "weka.core.DenseInstance"
include_class "weka.core.SparseInstance"




class SentimentsController < ApplicationController
  def index
  end
  
  def setup_classifier
    unless @classifier
      arff_path = Rails.root.join("data/sentiment.arff").to_s
      @arff = FileReader.new(arff_path)

      model_path = Rails.root.join("models/sentiment.model").to_s
      @classifier = SerializationHelper.read(model_path)

      @data = begin
        Instances.new(@arff,1).tap do |data|
          if data.class_index == -1
            data.setClassIndex data.num_attributes - 1
          end
        end
      end
    end
  end
  
  def create
    setup_classifier

    instance = SparseInstance.new(@data.num_attributes)
    instance.set_dataset(@data)
    instance.set_value(@data.attribute(0), params[:sentiment][:message])

    result = @classifier.distribution_for_instance(instance).first

    percent_positive = 1 - result.to_f

    @message = "The text is #{(percent_positive*100.0).round}% positive"
    
    render :action => 'index'
  end
end
