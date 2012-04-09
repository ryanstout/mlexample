package weka.classifiers.functions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import de.bwaldvogel.liblinear.SolverType;


public class LibLINEAROptionsTest {

    @Test
    public void testDefaultOptions() throws Exception {
        LibLINEAR liblinear = new LibLINEAR();
        List<String> options = new ArrayList<String>();
        liblinear.setOptions(options.toArray(new String[options.size()]));
        assertEquals(1.0, liblinear.getBias(), 0);
        assertEquals(0.001, liblinear.getEps(), 0);
        assertEquals(1.0, liblinear.getCost(), 0);
        assertFalse(liblinear.getProbabilityEstimates());
        assertFalse(liblinear.getNormalize());
        assertEquals("", liblinear.getWeights());
    }

    @Test
    public void testSetOptions() throws Exception {
        LibLINEAR liblinear = new LibLINEAR();
        String options = "-B 1.5 -C 100 -E 0.0001 -S 6 -P";
        liblinear.setOptions(options.split(" "));
        assertEquals(1.5, liblinear.getBias(), 0);
        assertEquals(100, liblinear.getCost(), 0);
        assertEquals(0.0001, liblinear.getEps(), 0);
        assertEquals(SolverType.L1R_LR, liblinear.getSolverType());
        assertTrue(liblinear.getProbabilityEstimates());
    }
}
