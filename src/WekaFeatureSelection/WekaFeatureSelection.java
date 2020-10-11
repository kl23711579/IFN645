package WekaFeatureSelection;

import java.util.ArrayList;
import java.util.Random;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.OneRAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

public class WekaFeatureSelection {
    public static void main(String[] args) throws Exception {
        WekaFeatureSelection wfs = new WekaFeatureSelection();
        DataSource source = new DataSource("/Users/d1w1rnna/Documents/work/weka-3-8-4/data/ionosphere.arff");
        Instances data = source.getDataSet();
        data.setClass(data.attribute("class"));
        // wfs.doFilterClassification(data);

        NaiveBayes nb = new NaiveBayes();
        CfsSubsetEval cfs = new CfsSubsetEval();
        BestFirst bf = new BestFirst();
        // wfs.doFilteredClassification(nb, data, cfs, bf);

        WrapperSubsetEval wrapper = new WrapperSubsetEval();
        wrapper.setClassifier(new J48());
        // wfs.doFilteredClassification(nb, data, wrapper, bf);

        Ranker ranker = new Ranker();
        ranker.setNumToSelect(7);

        // ArrayList<ASEvaluation> asevals = new ArrayList<ASEvaluation>();
        // asevals.add(new OneRAttributeEval());
        // asevals.add(new InfoGainAttributeEval());
        // asevals.add(new GainRatioAttributeEval());

        // for(int i = 0; i < asevals.size(); i++) {
        //     wfs.doFilteredClassification(nb, data, asevals.get(i), ranker);
        // }

        // wfs.doFilteredClassification(nb, data, new OneRAttributeEval(), ranker);
        int[] indices = wfs.doFeatureSelection(data, new OneRAttributeEval(), ranker);
        
        // wfs.doFilteredClassification(nb, data, new InfoGainAttributeEval(), ranker);
        // wfs.doFilteredClassification(nb, data, new GainRatioAttributeEval(), ranker);
    }

    public void doFilterClassification(Instances data) throws Exception{
        // try {
        //     DataSource source = new DataSource("/Users/d1w1rnna/Documents/work/weka-3-8-4/data/iris.arff");
        //     Instances data = source.getDataSet();
        // } catch(Exception e) {
        //     System.out.println("Error");
        //     e.printStackTrace();
        // }
        doASClassfier(new AttributeSelectedClassifier(), data);
   }

   public void doFilteredClassification(AbstractClassifier classifier, Instances data, ASEvaluation evaluation, ASSearch search) throws Exception{
        AttributeSelectedClassifier asc1 = new AttributeSelectedClassifier();
        asc1.setEvaluator(evaluation);
        asc1.setClassifier(classifier);
        asc1.setSearch(search);
        asc1.buildClassifier(data);

        Evaluation evaluate = new Evaluation(data);
        evaluate.crossValidateModel(asc1, data, 10, new Random(1));
        System.out.println(evaluate.correct()/evaluate.numInstances());

   }

   public int[] doFeatureSelection(Instances data, ASEvaluation evaluation, ASSearch search) throws Exception{
        AttributeSelection as = new AttributeSelection();
        as.setEvaluator(evaluation);
        as.setSearch(search);
        as.SelectAttributes(data);
        int[] indices = as.selectedAttributes();

        System.out.println(as.numberAttributesSelected());
        String[] s = new String[as.numberAttributesSelected()];
        for(int i = 0; i < indices.length-1; i++) {
            s[i] = data.attribute(indices[i]).name();
        }
        System.out.println(String.join("-", s));

        return indices;
   }

   public void doASClassfier(AbstractClassifier classifier, Instances data) throws Exception {
       classifier.buildClassifier(data);
       Evaluation evaluate = new Evaluation(data);
       evaluate.crossValidateModel(classifier, data, 10, new Random(1));
       System.out.println(evaluate.correct()/evaluate.numInstances());
   }
}