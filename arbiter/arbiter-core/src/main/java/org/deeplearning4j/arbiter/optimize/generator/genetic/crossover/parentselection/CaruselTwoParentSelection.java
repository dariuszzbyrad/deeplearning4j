/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;

import java.util.List;
import java.util.stream.Collectors;

/**
 * A parent selection behavior that returns two random parents.
 *
 * @author Alexandre Boulanger
 */
public class CaruselTwoParentSelection extends TwoParentSelection {

    private final RandomGenerator rng;

    public CaruselTwoParentSelection() {
        this(new SynchronizedRandomGenerator(new JDKRandomGenerator()));
    }

    /**
     * Use a supplied RandomGenerator
     *
     * @param rng An instance of RandomGenerator
     */
    public CaruselTwoParentSelection(RandomGenerator rng) {
        this.rng = rng;
    }

    /**
     * Selects two random parents
     *
     * @return An array of parents genes. The outer array are the parents, and the inner array are the genes.
     */
    @Override
    public double[][] selectParents() {
        double[][] parents = new double[2][];

        List<Double> scores = population.stream().map(c -> c.getFitness()).collect(Collectors.toList());
        double minFitnessValue = scores.stream().mapToDouble(c -> c).min().getAsDouble();
        if (minFitnessValue < 0) {
            double delta = minFitnessValue * -1;
            scores = scores.stream().map(s -> s + delta).collect(Collectors.toList());
        }
        double sumFitnessValue = scores.stream().mapToDouble(c -> c * 1000).sum();


        parents[0] = population.get(selectParent(scores, (int) sumFitnessValue)).getGenes();
        parents[1] = population.get(selectParent(scores, (int) sumFitnessValue)).getGenes();

        return parents;
    }

    private int selectParent(List<Double> scores, int sumFitnessValue) {
        int index = rng.nextInt(sumFitnessValue);
        long sum = 0;
        int i=0;
        while(sum < index ) {
            sum = (int) (sum + scores.get(i++) * 1000);
        }
        return Math.max(0,i-1);
    }
}
