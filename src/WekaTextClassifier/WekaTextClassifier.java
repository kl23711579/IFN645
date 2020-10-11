package WekaTextClassifier;

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stopwords.WordsFromFile;
import weka.core.tokenizers.AlphabeticTokenizer;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class WekaTextClassifier {
    public static void main(String[] args) throws Exception {
        // String path = "/Users/d1w1rnna/Documents/work/weka-3-8-4/data/movie_review.arff";
        String path = "data/text.arff";
        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();
        data.setClassIndex(1);
        data.setClass(data.attribute("score"));
        data.deleteAttributeAt(6);
        data.deleteAttributeAt(5);
        data.deleteAttributeAt(4);
        data.deleteAttributeAt(1);
        data.deleteAttributeAt(0);
        System.out.println(data.toSummaryString());
        

        StringToWordVector swFilter = new StringToWordVector();
        swFilter.setAttributeIndices("2");
        swFilter.setIDFTransform(true);
        swFilter.setTFTransform(true);
        swFilter.setDoNotOperateOnPerClassBasis(true);
        swFilter.setOutputWordCounts(true);
        swFilter.setStemmer(new LovinsStemmer());
        swFilter.setStopwordsHandler(new WordsFromFile());
        swFilter.setTokenizer(new AlphabeticTokenizer());
        swFilter.setWordsToKeep(100);

        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(swFilter);
        fc.setClassifier(new NaiveBayes());
        fc.buildClassifier(data);
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(fc, data, 10, new Random(1));
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        
    }

    public void doAllClassification(AbstractClassifier classifier, Instances data) {}
}