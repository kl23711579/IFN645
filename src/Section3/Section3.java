package Section3;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;

import moa.classifiers.trees.HoeffdingTree;
import moa.streams.ArffFileStream;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.AlphabeticTokenizer;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Section3 {
    public static void main(String[] args) throws Exception{
        Section3 s3 = new Section3();

        // config electricity.arff
        Instances ele_data = s3.getDataInstances("data/electricity.arff");
        ele_data.setClass(ele_data.attribute("class"));

        // config soil.arff
        Instances s_data = s3.getDataInstances("data/soil.arff");
        s_data.setClass(s_data.attribute("Cover_Type"));

        //config text.arff use 4 text
        Instances t_data = s3.getDataInstances("data/text.arff");
        t_data.setClass(t_data.attribute("score"));
        t_data.deleteAttributeAt(6);
        t_data.deleteAttributeAt(5);
        t_data.deleteAttributeAt(4);
        t_data.deleteAttributeAt(1);
        t_data.deleteAttributeAt(0);
        System.out.println(t_data.toSummaryString());

        // config wordvector for text.arff
        StringToWordVector swFilter = new StringToWordVector();
        // swFilter.setAttributeIndices("first-3,4-last");
        swFilter.setAttributeIndices("first-last");
        swFilter.setIDFTransform(true);
        swFilter.setTFTransform(true);
        swFilter.setDoNotOperateOnPerClassBasis(true);
        swFilter.setOutputWordCounts(true);
        swFilter.setStemmer(new LovinsStemmer());
        swFilter.setStopwordsHandler(new Rainbow());
        swFilter.setTokenizer(new AlphabeticTokenizer());
        swFilter.setWordsToKeep(100);

        // create filter for text.arff
        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(swFilter);
        fc.setClassifier(new NaiveBayes());

        // long starttime = System.nanoTime();
        // s3.doClassification(new NaiveBayes(), ele_data, new Evaluation(ele_data));
        // long endtime = System.nanoTime();
        // double elapsed = (endtime - starttime);
        // elapsed = elapsed/1000000000;
        // System.out.println("Time: " + elapsed);

        // starttime = System.nanoTime();
        // s3.doClassification(new NaiveBayes(), s_data, new Evaluation(s_data));
        // endtime = System.nanoTime();
        // elapsed = (endtime - starttime);
        // elapsed = elapsed/1000000000;
        // System.out.println("Time: " + elapsed);

        // starttime = System.nanoTime();
        // s3.doClassification(fc, t_data, new Evaluation(t_data));
        // endtime = System.nanoTime();
        // elapsed = (endtime - starttime);
        // elapsed = elapsed/1000000000;
        // System.out.println("Time: " + elapsed);

        // -1 means the last element
        // s3.doMoaClassification("data/electricity.arff", -1);
        // s3.doMoaClassification("data/soil.arff", 54);
        s3.doMoaClassification("data/text2.arff", 1);
        // s3.doMoaTextClassification("data/text.arff", 4);

        // swFilter.setInputFormat(t_data);
        // Instances new_text_data = Filter.useFilter(t_data, swFilter);
        // s3.doMoaTextClassification(new_text_data);

    }

    public Instances getDataInstances(String DataSourceName) throws Exception{
        DataSource datasource = new DataSource(DataSourceName);
        Instances data = datasource.getDataSet();
        return data;
    }

    public void doClassification(AbstractClassifier classifier, Instances data, Evaluation eval) throws Exception{
        classifier.buildClassifier(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));
        System.out.println("Correct: " + eval.correct());
    }

    public void doMoaClassification(String filename, int classIndex) throws Exception{
        int numberOfCorrectSampleInstances = 0;
        int totalNumberOfSampleInstances = 0;

        ArffFileStream ele_moa = new ArffFileStream(filename, classIndex);
        HoeffdingTree htree = new HoeffdingTree();
        htree.setModelContext(ele_moa.getHeader());
        htree.prepareForUse();

        boolean isTesting = true;
        while (ele_moa.hasMoreInstances()) {
            Instance data = ele_moa.nextInstance().getData();
            
            
            if (isTesting) {
                if(htree.correctlyClassifies(data)) {
                    numberOfCorrectSampleInstances++;
                }
            }
            totalNumberOfSampleInstances++;
            htree.trainOnInstance(data);
        }

        double accuracy = (double) numberOfCorrectSampleInstances/ (double) totalNumberOfSampleInstances;
        System.out.println(totalNumberOfSampleInstances);
        System.out.println(accuracy);
    }

    public void doMoaTextClassification(Instances text_data) throws Exception {
        int numberOfCorrectSampleInstances = 0;
        int totalNumberOfSampleInstances = 0;

        WekaToSamoaInstanceConverter wc = new WekaToSamoaInstanceConverter();
        Enumeration<weka.core.Instance> enum_Instance = text_data.enumerateInstances();

        // com.yahoo.labs.samoa.instances.Instances test = wc.samoaInstances(text_data);

        HoeffdingTree htree = new HoeffdingTree();
        // htree.setModelContext();
        htree.prepareForUse();

        boolean isTesting = true;
        while(enum_Instance.hasMoreElements()) {
            Instance data = wc.samoaInstance(enum_Instance.nextElement());
            if (isTesting) {
                if(htree.correctlyClassifies(data)) {
                    numberOfCorrectSampleInstances++;
                }
            }
            totalNumberOfSampleInstances++;
            htree.trainOnInstance(data);
        }

        double accuracy = (double) numberOfCorrectSampleInstances/ (double) totalNumberOfSampleInstances;
        System.out.println(totalNumberOfSampleInstances);
        System.out.println(accuracy);

    }
}
