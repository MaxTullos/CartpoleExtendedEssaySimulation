/**
 * Represents a single Cartpole physical instance and its AI brain.
 * The brain can be a fixed-topology NeuralNetwork or a NEATGenome.
 * Supports 1, 2, or 3 poles via SimulationContext.pendulumMode().
 */
public class CartAgent {
    public Brain brain;

    public double cartX = 400;
    public double cartVelocity = 0;

    public double poleAngle;
    public double poleAngularVelocity = 0;

    public double poleAngle2;
    public double poleAngularVelocity2 = 0;

    public double poleAngle3;
    public double poleAngularVelocity3 = 0;

    public boolean alive = true;
    public double fitness = 0;
    public boolean isElite = false;

    private SimulationContext context;
    private int framesDown = 0;

    public CartAgent(Brain brain, SimulationContext context) {
        this.brain = brain;
        this.context = context;
        double startRad = Math.PI + Math.toRadians(context.getCurrentGenAngleDeg());
        this.poleAngle  = startRad;
        this.poleAngle2 = startRad;
        this.poleAngle3 = startRad;
    }

    public void update() {
        if (!alive) return;

        int mode = context.pendulumMode();

        // 1. Prepare normalised inputs
        double normX     = (cartX - 400.0) / 400.0;
        double normCartV = cartVelocity / 500.0;
        double normAng1  = (poleAngle  - Math.PI) / (Math.PI / 2.0);
        double normVel1  = poleAngularVelocity  / 5.0;

        double[] inputs;
        if (mode == 3) {
            double normAng2 = (poleAngle2 - Math.PI) / (Math.PI / 2.0);
            double normVel2 = poleAngularVelocity2 / 5.0;
            double normAng3 = (poleAngle3 - Math.PI) / (Math.PI / 2.0);
            double normVel3 = poleAngularVelocity3 / 5.0;
            inputs = new double[]{normX, normCartV, normAng1, normVel1, normAng2, normVel2, normAng3, normVel3};
        } else if (mode == 2) {
            double normAng2 = (poleAngle2 - Math.PI) / (Math.PI / 2.0);
            double normVel2 = poleAngularVelocity2 / 5.0;
            inputs = new double[]{normX, normCartV, normAng1, normVel1, normAng2, normVel2};
        } else {
            inputs = new double[]{normX, normCartV, normAng1, normVel1};
        }

        // 2. Get brain decision
        double[] outputs = brain.feedForward(inputs);
        double force = outputs[0];

        // 3. Apply cart physics
        double cartAccelPixels = force * 400.0;
        cartVelocity += cartAccelPixels * Config.DT;
        cartVelocity *= 0.95;
        cartX += cartVelocity * Config.DT;

        double a = cartAccelPixels / Config.PIXELS_PER_METER; // cart acceleration in m/s²

        double l = Config.PHYSICAL_POLE_LENGTH;
        double g = Config.GRAVITY;

        if (mode == 3) {
            // ── Triple Pendulum (cart-driven Lagrangian) ─────────────────────
            // Masses and lengths (uniform, each = 1 kg, shared pole length l)
            double m1 = 1.0, m2 = 1.0, m3 = 1.0;

            double th1 = poleAngle,  om1 = poleAngularVelocity;
            double th2 = poleAngle2, om2 = poleAngularVelocity2;
            double th3 = poleAngle3, om3 = poleAngularVelocity3;

            double d12 = th1 - th2;
            double d13 = th1 - th3;
            double d23 = th2 - th3;

            // Mass matrix (3×3, symmetric)
            double M11 = (m1 + m2 + m3) * l;
            double M12 = (m2 + m3) * l * Math.cos(d12);
            double M13 = m3 * l * Math.cos(d13);
            double M21 = M12;
            double M22 = (m2 + m3) * l;
            double M23 = m3 * l * Math.cos(d23);
            double M31 = M13;
            double M32 = M23;
            double M33 = m3 * l;

            // Generalised force vector
            double F1 = -(m1 + m2 + m3) * g * Math.sin(th1)
                       - (m1 + m2 + m3) * a * Math.cos(th1)
                       - (m2 + m3) * l * om2 * om2 * Math.sin(d12)
                       - m3 * l * om3 * om3 * Math.sin(d13)
                       - 0.05 * om1;

            double F2 = -(m2 + m3) * g * Math.sin(th2)
                       - (m2 + m3) * a * Math.cos(th2)
                       + (m2 + m3) * l * om1 * om1 * Math.sin(d12)
                       - m3 * l * om3 * om3 * Math.sin(d23)
                       - 0.05 * om2;

            double F3 = -m3 * g * Math.sin(th3)
                       - m3 * a * Math.cos(th3)
                       + m3 * l * om1 * om1 * Math.sin(d13)
                       + m3 * l * om2 * om2 * Math.sin(d23)
                       - 0.05 * om3;

            // Solve M * alpha = F via Cramer's rule
            double det = M11 * (M22 * M33 - M23 * M32)
                       - M12 * (M21 * M33 - M23 * M31)
                       + M13 * (M21 * M32 - M22 * M31);

            double alpha1 = (F1 * (M22 * M33 - M23 * M32)
                           - M12 * (F2 * M33 - M23 * F3)
                           + M13 * (F2 * M32 - M22 * F3)) / det;

            double alpha2 = (M11 * (F2 * M33 - M23 * F3)
                           - F1 * (M21 * M33 - M23 * M31)
                           + M13 * (M21 * F3 - F2 * M31)) / det;

            double alpha3 = (M11 * (M22 * F3 - F2 * M32)
                           - M12 * (M21 * F3 - F2 * M31)
                           + F1 * (M21 * M32 - M22 * M31)) / det;

            poleAngularVelocity  = (om1 + alpha1 * Config.DT) * 0.99;
            poleAngularVelocity2 = (om2 + alpha2 * Config.DT) * 0.99;
            poleAngularVelocity3 = (om3 + alpha3 * Config.DT) * 0.99;

            poleAngle  = normalise(poleAngle  + poleAngularVelocity  * Config.DT);
            poleAngle2 = normalise(poleAngle2 + poleAngularVelocity2 * Config.DT);
            poleAngle3 = normalise(poleAngle3 + poleAngularVelocity3 * Config.DT);

        } else if (mode == 2) {
            // ── Double Pendulum (original Lagrangian) ─────────────────────────
            double m1 = 1.0, m2 = 1.0;
            double d12 = poleAngle - poleAngle2;

            double M11 = (m1 + m2) * l;
            double M12 = m2 * l * Math.cos(d12);
            double M21 = m2 * l * Math.cos(d12);
            double M22 = m2 * l;

            double f1 = -(m1 + m2) * g * Math.sin(poleAngle)
                        - (m1 + m2) * a * Math.cos(poleAngle)
                        - m2 * l * poleAngularVelocity2 * poleAngularVelocity2 * Math.sin(d12)
                        - 0.05 * poleAngularVelocity;

            double f2 = -m2 * g * Math.sin(poleAngle2)
                        - m2 * a * Math.cos(poleAngle2)
                        + m2 * l * poleAngularVelocity * poleAngularVelocity * Math.sin(d12)
                        - 0.05 * poleAngularVelocity2;

            double D = M11 * M22 - M12 * M21;

            poleAngularVelocity  = (poleAngularVelocity  + (f1 * M22 - f2 * M12) / D * Config.DT) * 0.99;
            poleAngularVelocity2 = (poleAngularVelocity2 + (f2 * M11 - f1 * M21) / D * Config.DT) * 0.99;

            poleAngle  = normalise(poleAngle  + poleAngularVelocity  * Config.DT);
            poleAngle2 = normalise(poleAngle2 + poleAngularVelocity2 * Config.DT);

        } else {
            // ── Single Pendulum ───────────────────────────────────────────────
            double alpha = (-a * Math.cos(poleAngle) - g * Math.sin(poleAngle)) / l;
            poleAngularVelocity = (poleAngularVelocity + alpha * Config.DT) * 0.99;
            poleAngle = normalise(poleAngle + poleAngularVelocity * Config.DT);
        }

        // 4. Check death conditions
        boolean died = false;

        if (!context.useAlternativeFitness()) {
            died |= Math.abs(poleAngle - Math.PI) > Math.PI / 2.0;
            if (mode >= 2) died |= Math.abs(poleAngle2 - Math.PI) > Math.PI / 2.0;
            if (mode == 3) died |= Math.abs(poleAngle3 - Math.PI) > Math.PI / 2.0;
        } else {
            double currentY = -Math.cos(poleAngle);
            if (mode >= 2) currentY += -Math.cos(poleAngle2);
            if (mode == 3) currentY += -Math.cos(poleAngle3);
            currentY /= mode; // average over all poles

            if (currentY < Config.downYThreshold) {
                framesDown++;
                if (framesDown > (Config.maxDownTime / Config.DT)) died = true;
            } else {
                framesDown = 0;
            }
        }

        if (died) {
            alive = false;
        } else {
            // 5. Calculate fitness
            if (context.useAlternativeFitness()) {
                double poleEndX = cartX + Math.sin(poleAngle) * Config.POLE_LENGTH_PX;
                double poleEndYRel = -Math.cos(poleAngle);
                if (mode >= 2) {
                    poleEndX += Math.sin(poleAngle2) * Config.POLE_LENGTH_PX;
                    poleEndYRel += -Math.cos(poleAngle2);
                }
                if (mode == 3) {
                    poleEndX += Math.sin(poleAngle3) * Config.POLE_LENGTH_PX;
                    poleEndYRel += -Math.cos(poleAngle3);
                }
                poleEndYRel /= mode; // normalise

                double heightScore  = Math.max(0, poleEndYRel);
                double heightReward = (Math.exp(heightScore * 2.0) - 1.0) * 10.0;
                double centerScore  = 0;
                if (context.useCenterReward()) {
                    centerScore = 1.0 - Math.min(1.0, Math.abs(poleEndX - 400.0) / 400.0);
                }
                fitness += heightReward + centerScore;
            } else {
                double fitnessIncrement = 1.0;
                if (context.useCenterReward()) {
                    fitnessIncrement -= (Math.abs(cartX - 400.0) / 400.0) * 0.5;
                }
                fitness += fitnessIncrement;
            }
        }
    }

    private static double normalise(double angle) {
        angle = angle % (2 * Math.PI);
        if (angle < 0) angle += 2 * Math.PI;
        return angle;
    }
}
