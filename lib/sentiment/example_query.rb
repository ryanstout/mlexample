# Example of how to query a model that is already built

arff_path = Rails.root.join("data/sentiment.arff").to_s
arff = FileReader.new(arff_path)

model_path = Rails.root.join("models/sentiment.model").to_s
classifier = SerializationHelper.read(model_path)

data = begin
  Instances.new(@arff,1).tap do |instance|
    if instance.class_index == -1
      instance.set_class_index(instance.num_attributes - 1)
    end
  end
end

instance = SparseInstance.new(data.num_attributes)
instance.set_dataset(data)
instance.set_value(data.attribute(0), params[:sentiment][:message])

result = classifier.distribution_for_instance(instance).first

percent_positive = 1 - result.to_f
