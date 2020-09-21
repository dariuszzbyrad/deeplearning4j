package org.deeplearning4j.arbiter.optimize;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.arbiter.optimize.api.*;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.CandidateInfo;
import org.deeplearning4j.arbiter.optimize.runner.CandidateStatus;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.listener.StatusListener;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ShekelFunction {

	public static class ShekelSpace extends AbstractParameterSpace<ShekelConfig> {
		private static List<ParameterSpace<Double>> parameterSpaces;

		static {
			parameterSpaces = IntStream.range(0, 4)
				.mapToObj(i -> new ContinuousParameterSpace(-10, 10))
				.collect(Collectors.toList());
		}

		@Override
		public ShekelConfig getValue(double[] parameterValues) {
			List<Double> values = parameterSpaces.stream()
				.map(ps -> ps.getValue(parameterValues))
				.collect(Collectors.toList());

			return new ShekelConfig(values);
		}

		@Override
		public int numParameters() {
			return 4;
		}

		@Override
		public List<ParameterSpace> collectLeaves() {
			return parameterSpaces.stream()
				.map(ps -> ps.collectLeaves())
				.flatMap(List::stream)
				.collect(Collectors.toList());
		}

		@Override
		public boolean isLeaf() {
			return false;
		}

		@Override
		public void setIndices(int... indices) {
			throw new UnsupportedOperationException();
		}
	}

	@AllArgsConstructor
	@Data
	public static class ShekelConfig implements Serializable {
		private List<Double> xn;
	}

	public static class ShekelScoreFunction implements ScoreFunction {
		private static final int LOCAL_MINIMA = 10;
		private static final int DIMENSIONS = 4;

		private static final double[] B = new double[]{0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5};

		private static final double[][] C = new double[][]{
			new double[]{4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0},
			new double[]{4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6},
			new double[]{4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0},
			new double[]{4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6}
		};

		@Override
		public double score(Object m, DataProvider data, Map<String, Object> dataParameters) {
			ShekelConfig model = (ShekelConfig) m;
			List<Double> xn = model.getXn();
			double result = 0;
			for (int i = 0; i < LOCAL_MINIMA; i++) {

				double partialResult = 0;
				for (int j = 0; j < DIMENSIONS; j++) {
					partialResult += Math.pow(xn.get(j) - C[j][i], 2);
				}
				partialResult += B[i];
				result += (1 / partialResult);
			}

			return result * -1;
		}

		@Override
		public double score(Object model, Class<? extends DataSource> dataSource, Properties dataSourceProperties) {
			throw new UnsupportedOperationException();
		}

		@Override
		public boolean minimize() {
			return true;
		}

		@Override
		public List<Class<?>> getSupportedModelTypes() {
			return Collections.<Class<?>>singletonList(ShekelConfig.class);
		}

		@Override
		public List<Class<?>> getSupportedDataTypes() {
			return Collections.<Class<?>>singletonList(Object.class);
		}
	}

	public static class ShekelTaskCreator implements TaskCreator {
		@Override
		public Callable<OptimizationResult> create(final Candidate c, DataProvider dataProvider,
												   final ScoreFunction scoreFunction, final List<StatusListener> statusListeners,
												   IOptimizationRunner runner) {

			return () -> {
				ShekelConfig candidate = (ShekelConfig) c.getValue();

				double score = scoreFunction.score(candidate, null, (Map) null);

				Thread.sleep(20);

				if (statusListeners != null) {
					for (StatusListener sl : statusListeners) {
						sl.onCandidateIteration(null, null, 0);
					}
				}

				CandidateInfo ci = new CandidateInfo(-1, CandidateStatus.Complete, score,
					System.currentTimeMillis(), null, null, null, null);

				return new OptimizationResult(c, score, c.getIndex(), null, ci, null);
			};
		}

		@Override
		public Callable<OptimizationResult> create(Candidate candidate, Class<? extends DataSource> dataSource,
												   Properties dataSourceProperties, ScoreFunction scoreFunction,
												   List<StatusListener> statusListeners, IOptimizationRunner runner) {
			throw new UnsupportedOperationException();
		}
	}
}
