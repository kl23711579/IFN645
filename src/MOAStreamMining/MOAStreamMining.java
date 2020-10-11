package MOAStreamMining;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.trees.HoeffdingTree;
import moa.streams.generators.RandomRBFGenerator;

public class MOAStreamMining {
    public static void main(String[] args) throws Exception {
        int numberOfInstances = 100000;
        int numberOfCorrectSampleInstances = 0;
        int totalNumberOfSampleInstances = 0;

        RandomRBFGenerator rrg = new RandomRBFGenerator();
        rrg.prepareForUse();

        HoeffdingTree htree = new HoeffdingTree();
        htree.setModelContext(rrg.getHeader());
        htree.prepareForUse();

        boolean isTesting = true;
        while (rrg.hasMoreInstances() && totalNumberOfSampleInstances < numberOfInstances) {
            Instance data = rrg.nextInstance().getData();
            
            
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