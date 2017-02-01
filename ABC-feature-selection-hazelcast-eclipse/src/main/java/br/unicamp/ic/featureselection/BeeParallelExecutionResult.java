package br.unicamp.ic.featureselection;

import java.io.Serializable;
import java.util.Set;

import br.unicamp.ic.featureselection.swarm.FoodSource;

public class BeeParallelExecutionResult implements Serializable {

	private static final long serialVersionUID = -6082076998889253031L;

	/**
	 * Armazena os as fontes que não serão mais verificadas para serem removidas
	 * logo em seguida
	 */
	private Set<FoodSource> markedToRemoved;

	/**
	 * Armazena as fontes vizinhas que serão consideradas na busca
	 */
	private Set<FoodSource> neighbors;

	/**
	 * Armazena as fontes abandonadas para posteriormente removê-las como fonte
	 * de alimento
	 */
	private Set<FoodSource> abandoned;

	/**
	 * Fontes de alimento que já foram visitadas
	 */
	private Set<FoodSource> visitedFoodSources;

	private FoodSource bestFoodSource;

	private double bestFitness;

	public BeeParallelExecutionResult(Set<FoodSource> markedToRemoved,
			Set<FoodSource> neighbors, Set<FoodSource> abandoned,
			Set<FoodSource> visitedFoodSources, FoodSource bestFoodSource,
			double bestFitness) {

		this.markedToRemoved = markedToRemoved;
		this.neighbors = neighbors;
		this.abandoned = abandoned;
		this.visitedFoodSources = visitedFoodSources;
		this.bestFoodSource = bestFoodSource;
		this.bestFitness = bestFitness;
	}

	public Set<FoodSource> getMarkedToRemoved() {
		return markedToRemoved;
	}

	public void setMarkedToRemoved(Set<FoodSource> markedToRemoved) {
		this.markedToRemoved = markedToRemoved;
	}

	public Set<FoodSource> getNeighbors() {
		return neighbors;
	}

	public void setNeighbors(Set<FoodSource> neighbors) {
		this.neighbors = neighbors;
	}

	public Set<FoodSource> getAbandoned() {
		return abandoned;
	}

	public void setAbandoned(Set<FoodSource> abandoned) {
		this.abandoned = abandoned;
	}

	public Set<FoodSource> getVisitedFoodSources() {
		return visitedFoodSources;
	}

	public void setVisitedFoodSources(Set<FoodSource> visitedFoodSources) {
		this.visitedFoodSources = visitedFoodSources;
	}

	public FoodSource getBestFoodSource() {
		return bestFoodSource;
	}

	public void setBestFoodSource(FoodSource bestFoodSource) {
		this.bestFoodSource = bestFoodSource;
	}

	public double getBestFitness() {
		return bestFitness;
	}

	public void setBestFitness(double bestFitness) {
		this.bestFitness = bestFitness;
	}
}
