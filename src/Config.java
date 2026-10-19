public class Config {
    // Simulation Constants
    public static final double GRAVITY = 9.81;
    public static final double PIXELS_PER_METER = 150.0;
    public static final double PHYSICAL_POLE_LENGTH = 1.0;
    public static final double POLE_LENGTH_PX = PHYSICAL_POLE_LENGTH * PIXELS_PER_METER;
    public static final double DT = 0.016; // ~60 FPS
    public static final int CART_WIDTH = 80;
    public static final int CART_HEIGHT = 40;

    // Death conditions for alternative fitness
    public static double maxDownTime = 2.0; // Seconds before dying if fully down
    public static double downYThreshold = -0.9; // Y-relative value below which is considered 'down'
}
