public interface SimulationContext {
    /** 1 = single, 2 = double, 3 = triple */
    int pendulumMode();
    boolean useAlternativeFitness();
    boolean useCenterReward();
    double getCurrentGenAngleDeg();

    /** Convenience: true when running 2 or more poles. */
    default boolean isDoublePendulum() { return pendulumMode() >= 2; }
}
