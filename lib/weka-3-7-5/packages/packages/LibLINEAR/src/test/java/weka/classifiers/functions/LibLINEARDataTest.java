package weka.classifiers.functions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import de.bwaldvogel.liblinear.SolverType;


public class LibLINEARDataTest {

    private final File TEST_RESOURCES = new File("src/test/resources");

    private int doCrossValidation(Instances insts, Classifier classifier, final int FOLDS) throws Exception {
        int errors = 0;

        for (int fold = 0; fold < FOLDS; fold++) {
            classifier.buildClassifier(insts.trainCV(FOLDS, fold));
            Instances testInstances = insts.testCV(FOLDS, fold);
            for (Instance testInstance : testInstances) {
                double result = classifier.classifyInstance(testInstance);
                double value = testInstance.classValue();
                if (value != result) {
                    errors++;
                }
            }
        }

        return errors;
    }

    @Test
    public void testWeather() throws Exception {
        Instances insts = loadInstancesFromARFF("weather.arff", "play");

        LibLINEAR liblinear = new LibLINEAR();
        for (SolverType solverType : new SolverType[] {SolverType.L2R_LR, SolverType.L1R_LR}) {
            liblinear.setSolverType(solverType);
            liblinear.setBias(-1);
            liblinear.setCost(1000); // explicitly high costs to reduce training error
            liblinear.setEps(1e-3);
            liblinear.setProbabilityEstimates(true);

            int errors = 0;
            liblinear.buildClassifier(insts);
            for (Instance instance : insts) {
                double[] d = liblinear.distributionForInstance(instance);
                assertEquals(2, d.length);
                int prediction = (int)liblinear.classifyInstance(instance);
                assertTrue(d[prediction] > d[1 - prediction]);
                if (prediction != (int)instance.classValue()) {
                    errors++;
                }
            }
            assertEquals(3, errors);
        }
    }

    @Test
    public void testIris() throws Exception {
        Instances insts = loadInstancesFromARFF("iris.arff", "class");

        LibLINEAR liblinear = new LibLINEAR();
        liblinear.setSolverType(SolverType.MCSVM_CS);
        liblinear.setBias(1);
        liblinear.setCost(1);
        liblinear.setEps(1e-2);
        liblinear.setProbabilityEstimates(false);

        int errors = doCrossValidation(insts, liblinear, 10);
        assertEquals(6, errors);
    }

    private Instances loadInstancesFromARFF(String filename, String className) throws IOException {
        File file = new File(TEST_RESOURCES, filename);
        assertTrue(file.exists());

        ArffLoader loader = new ArffLoader();
        loader.setFile(file);
        Instances insts = loader.getDataSet();
        Attribute classAttribute = insts.attribute(className);
        insts.setClass(classAttribute);
        return insts;
    }


    // tests for bug that was reported by Ondrej Dusek
    @Test
    public void testIrisNoVirginica() throws Exception {
        Instances insts = loadInstancesFromARFF("iris-novirginica.arff", "class");

        LibLINEAR liblinear = new LibLINEAR();
        liblinear.setSolverType(SolverType.L2R_LR);
        liblinear.setProbabilityEstimates(true);
        liblinear.buildClassifier(insts);

        for (Instance instance : insts) {
            liblinear.classifyInstance(instance);
            liblinear.distributionForInstance(instance);
        }

    }
}
