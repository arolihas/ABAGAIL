package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class ContinuousPeaksTest {
    /** The n value */
    //private static final int N = 60;
    /** The t value */
    //private static final int T = N / 10;
    
    public static void main(String[] args) {
	if (args.length < 2) {
		System.out.println("Provide a input size and repeat counter");
	        System.exit(0);
	}
	int N = Integer.parseInt(args[0]);
	int T = N/10;
	int iterations = Integer.parseInt(args[1]);
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new ContinuousPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
     	System.out.println("RHC");
	for (int i = 0; i < iterations; i++) {
		RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
		long t = System.nanoTime();
		FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 20000);
		fit.train();
		System.out.println(ef.value(rhc.getOptimal()) + ", " + 
		(((double)(System.nanoTime() - t))/ 1e9d));
        }
	
	System.out.println("SA");
	for (int i = 0; i < iterations; i++) {
		SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);	
		long t = System.nanoTime();
		FixedIterationTrainer fit = new FixedIterationTrainer(sa, 20000);
		fit.train();
		System.out.println(ef.value(sa.getOptimal()) + ", " + 
		(((double)(System.nanoTime() - t))/ 1e9d));
	}
	System.out.println("GA");
        for (int i = 0; i < iterations; i++) {
		StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 10, 1, gap);
		long t = System.nanoTime();        
		FixedIterationTrainer fit = new FixedIterationTrainer(ga, 10000);
		fit.train();
		System.out.println(ef.value(ga.getOptimal())  + ", " +
		(((double)(System.nanoTime() - t))/ 1e9d));
	}
    }
}
