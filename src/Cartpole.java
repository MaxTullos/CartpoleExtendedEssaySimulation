import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.List;
import java.util.HashMap;

/**
 * Cartpole Simulation controlled by a Neural Network using a Genetic Algorithm.
 */
public class Cartpole extends JPanel implements ActionListener, KeyListener, SimulationContext {

    private int maxFramesPerGen = 1800; // ~30 seconds max per generation

    private Timer timer;
    private ArrayList<CartAgent> population;
    private int generation = 1;
    private int framesThisGen = 0;
    private double bestFitnessEver = 0;
    private double finalBestFitness = 0;
    private double finalMeanFitness = 0;
    private double finalMedianFitness = 0;
    private double finalStdDevFitness = 0;

    private double cameraZoom = 1.0;
    private double cameraX = 400.0;

    // Added fields for training control and replay
    private int targetGenerations = 30;
    private double startingAngleDeg = 5.0;
    private double angleVarianceDeg = 0.0;
    private double currentGenAngleDeg = 5.0;
    private int pendulumMode = 1; // 1 = single, 2 = double, 3 = triple
    
    // Configurable Genetic Algorithm Parameters
    private int populationSize = 150;
    private double elitismPercent = 5.0;
    private double mutationRate = 5.0;
    private double mutationMagnitude = 0.5;
    private double ghostRenderPercent = 100.0;
    private int tournamentSize = 4;
    private boolean useAlternativeFitness = false;
    private boolean useCenterReward = true;
    private boolean fastForwardChildren = true;
    private boolean showHistogram = false;
    private boolean showNN = true;

    // ── NEAT mode ─────────────────────────────────────────────────────────────
    private boolean useNEAT = false;
    private ArrayList<NEATSpecies> neatSpecies = new ArrayList<>();
    private NEATInnovationTracker neatTracker = null;
    // NEAT hyperparameters
    private double neatCompatThreshold  = 3.0;
    private double neatAddNodeRate      = 0.03;
    private double neatAddConnRate      = 0.05;
    private double neatWeightMutRate    = 0.80;
    private double neatCrossoverRate    = 0.75;
    private double neatSurvivalFraction = 0.20;
    private int    neatStagnationLimit  = 15;
    
    // Add this line back in!
    private double[] previousGenerationFitness = null;

    private enum SimulationState { TRAINING, WAITING_FOR_REPLAY, REPLAY }
    private SimulationState simState = SimulationState.TRAINING;
    
    private Brain globalBestBrain = null;
    private CartAgent replayAgent = null;
    private ArrayList<GenerationData> performanceHistory = new ArrayList<>();

    private JCheckBox renderCheckBox;
    private JButton startReplayBtn;
    private JButton testNewAngleBtn;
    private JButton continueTrainingBtn;
    private JButton exportBtn;
    private JButton importBtn;
    private JFrame liveGraphFrame = null;
    private JSlider thicknessSlider;
    private JLabel thicknessLabel;
    private JSlider simSpeedSlider;
    private JLabel simSpeedLabel;

    // Neural Network settings requested by user
    private String hiddenLayerStr = "6";
    private int[] hiddenLayerSizes = {6};

    public Cartpole() {
        promptForSettings();

        setBackground(new Color(240, 240, 240));
        setFocusable(true);
        setLayout(null); // Use absolute positioning for the button

        startReplayBtn = new JButton("Start Replay");
        startReplayBtn.setBounds(300, 250, 200, 50);
        startReplayBtn.setFont(new Font("SansSerif", Font.BOLD, 18));
        startReplayBtn.setVisible(false);
        startReplayBtn.addActionListener(e -> {
            startReplayBtn.setVisible(false);
            testNewAngleBtn.setVisible(false);
            continueTrainingBtn.setVisible(false);
            simState = SimulationState.REPLAY;
            startReplay();
        });
        add(startReplayBtn);

        testNewAngleBtn = new JButton("Test New Angle");
        testNewAngleBtn.setBounds(20, 180, 200, 30);
        testNewAngleBtn.setFont(new Font("SansSerif", Font.BOLD, 14));
        testNewAngleBtn.setVisible(false);
        testNewAngleBtn.addActionListener(e -> {
            String input = JOptionPane.showInputDialog(this, "Enter new starting angle (Degrees from top):", String.valueOf(startingAngleDeg));
            if (input != null) {
                try {
                    startingAngleDeg = Double.parseDouble(input);
                    currentGenAngleDeg = startingAngleDeg; // For a direct test, use the exact angle
                    
                    // Switch back to TRAINING mode to reuse the fast-forward physics and original overlay
                    simState = SimulationState.TRAINING;
                    startReplayBtn.setVisible(false); 
                    testNewAngleBtn.setVisible(false);
                    continueTrainingBtn.setVisible(false);
                    
                    // Reset the current population in-place to start fresh at the new angle
                    for (CartAgent agent : population) {
                        agent.cartX = 400;
                        agent.cartVelocity = 0;
                        agent.poleAngle = Math.PI + Math.toRadians(currentGenAngleDeg);
                        agent.poleAngle2 = Math.PI + Math.toRadians(currentGenAngleDeg);
                        agent.poleAngularVelocity = 0;
                        agent.poleAngularVelocity2 = 0;
                        agent.alive = true;
                        agent.fitness = 0;
                    }
                    framesThisGen = 0;
                    
                } catch (NumberFormatException ex) {
                    JOptionPane.showMessageDialog(this, "Invalid number.");
                }
            }
        });
        add(testNewAngleBtn);

        continueTrainingBtn = new JButton("Continue Training");
        continueTrainingBtn.setBounds(20, 215, 200, 30);
        continueTrainingBtn.setFont(new Font("SansSerif", Font.BOLD, 14));
        continueTrainingBtn.setVisible(false);
        continueTrainingBtn.addActionListener(e -> {
            JTextField gensField = new JTextField("10");
            JTextField angleField = new JTextField(String.valueOf(startingAngleDeg));
            Object[] message = {
                "Enter additional generations to train:", gensField,
                "New base starting angle (Degrees from top):", angleField
            };
            
            int option = JOptionPane.showConfirmDialog(this, message, "Continue Training", JOptionPane.OK_CANCEL_OPTION);
            if (option == JOptionPane.OK_OPTION) {
                try {
                    int addGens = Integer.parseInt(gensField.getText());
                    if (addGens > 0) {
                        targetGenerations += addGens;
                        startingAngleDeg = Double.parseDouble(angleField.getText());
                        
                        simState = SimulationState.TRAINING;
                        startReplayBtn.setVisible(false);
                        testNewAngleBtn.setVisible(false);
                        continueTrainingBtn.setVisible(false);
                        
                        // Immediately breed the current generation and resume training
                        evolveNextGeneration();
                    }
                } catch (NumberFormatException ex) {
                    JOptionPane.showMessageDialog(this, "Invalid input. Please enter valid numbers.");
                }
            }
        });
        add(continueTrainingBtn);

        thicknessLabel = new JLabel("NN Line Thickness");
        thicknessLabel.setFont(new Font("SansSerif", Font.BOLD, 12));
        thicknessLabel.setForeground(Color.DARK_GRAY);
        add(thicknessLabel);

        thicknessSlider = new JSlider(1, 100, 15); // Represents scales from 0.1x to 10.0x
        thicknessSlider.setOpaque(false);
        add(thicknessSlider);

        simSpeedLabel = new JLabel("Simulation Speed (1x - 1000x)");
        simSpeedLabel.setFont(new Font("SansSerif", Font.BOLD, 12));
        simSpeedLabel.setForeground(Color.DARK_GRAY);
        add(simSpeedLabel);

        simSpeedSlider = new JSlider(1, 1000, 5); // Default speed: 5x, Max: 1000x
        simSpeedSlider.setOpaque(false);
        simSpeedSlider.setFocusable(false); // Prevents stealing keyboard focus
        add(simSpeedSlider);

        exportBtn = new JButton("Export AI Brain");
        exportBtn.setFont(new Font("SansSerif", Font.BOLD, 12));
        exportBtn.setFocusable(false);
        exportBtn.addActionListener(e -> exportBrain());
        add(exportBtn);

        importBtn = new JButton("Import AI Brain");
        importBtn.setFont(new Font("SansSerif", Font.BOLD, 12));
        importBtn.setFocusable(false);
        importBtn.addActionListener(e -> importBrain());
        add(importBtn);

        JButton viewGraphBtn = new JButton("View Performance Graph");
        viewGraphBtn.setFont(new Font("SansSerif", Font.BOLD, 12));
        viewGraphBtn.setFocusable(false);
        viewGraphBtn.addActionListener(e -> {
            if (liveGraphFrame == null || !liveGraphFrame.isVisible()) {
                liveGraphFrame = PerformanceGraph.showGraph(performanceHistory, previousGenerationFitness);
            } else {
                liveGraphFrame.toFront();
            }
        });
        add(viewGraphBtn);

        JButton exportCsvBtn = new JButton("Export Metrics (CSV)");
        exportCsvBtn.setFont(new Font("SansSerif", Font.BOLD, 12));
        exportCsvBtn.setFocusable(false);
        exportCsvBtn.addActionListener(e -> exportMetricsToCSV());
        add(exportCsvBtn);

        renderCheckBox = new JCheckBox("Disable Rendering (Faster)", false);
        renderCheckBox.setFont(new Font("SansSerif", Font.BOLD, 12));
        renderCheckBox.setOpaque(false);
        renderCheckBox.setFocusable(false);
        add(renderCheckBox);

        addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                // Dynamically reposition the slider below the NN visualization
                int nnWidth = Math.max(200, getWidth() / 4);
                int nnHeight = Math.max(150, getHeight() / 3);
                int nnStartX = getWidth() - nnWidth - 80;
                int nnStartY = 50;
                thicknessLabel.setBounds(nnStartX, nnStartY + nnHeight + 10, nnWidth, 20);
                thicknessSlider.setBounds(nnStartX, nnStartY + nnHeight + 30, nnWidth, 30);
                simSpeedLabel.setBounds(nnStartX, nnStartY + nnHeight + 60, nnWidth, 20);
                simSpeedSlider.setBounds(nnStartX, nnStartY + nnHeight + 80, nnWidth, 30);
                exportBtn.setBounds(nnStartX, nnStartY + nnHeight + 120, nnWidth, 30);
                importBtn.setBounds(nnStartX, nnStartY + nnHeight + 160, nnWidth, 30);
                viewGraphBtn.setBounds(nnStartX, nnStartY + nnHeight + 200, nnWidth, 30);
                exportCsvBtn.setBounds(nnStartX, nnStartY + nnHeight + 240, nnWidth, 30);
                renderCheckBox.setBounds(nnStartX, nnStartY + nnHeight + 280, nnWidth, 30);
            }
        });

        addKeyListener(this);

        initializePopulation();

        // Timer for physics and rendering updates
        timer = new Timer(16, this);
        timer.start();
    }

    private void promptForSettings() {
        JTextField topologyField      = new JTextField(hiddenLayerStr, 10);
        JTextField gensField          = new JTextField("30", 10);
        JTextField maxFramesField     = new JTextField("1800", 10);
        JTextField angleField         = new JTextField("5.0", 10);
        JTextField varianceField      = new JTextField("0.0", 10);
        JTextField popField           = new JTextField("150", 10);
        JTextField elitismField       = new JTextField("5.0", 10);
        JTextField mutationField      = new JTextField("5.0", 10);
        JTextField mutMagField        = new JTextField("0.5", 10);
        JTextField ghostPercentField  = new JTextField("100.0", 10);
        JTextField tourneyField       = new JTextField("4", 10);
        JTextField maxDownTimeField   = new JTextField(String.valueOf(Config.maxDownTime), 10);
        JTextField downThresholdField = new JTextField(String.valueOf(Config.downYThreshold), 10);

        // Pendulum mode radio buttons
        JRadioButton singlePole = new JRadioButton("Single",  pendulumMode == 1);
        JRadioButton doublePole = new JRadioButton("Double",  pendulumMode == 2);
        JRadioButton triplePole = new JRadioButton("Triple",  pendulumMode == 3);
        ButtonGroup poleGroup = new ButtonGroup();
        poleGroup.add(singlePole); poleGroup.add(doublePole); poleGroup.add(triplePole);
        JPanel polePanel = new JPanel(new GridLayout(1, 3));
        polePanel.add(singlePole); polePanel.add(doublePole); polePanel.add(triplePole);
        polePanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Pendulum Mode"));
        polePanel.setAlignmentX(Component.LEFT_ALIGNMENT);

        JCheckBox altFitnessCheck      = new JCheckBox("Use Alt Fitness (Pole End Y & Center)");
        JCheckBox useCenterRewardCheck = new JCheckBox("Reward Center Proximity", true);
        JCheckBox fastForwardCheck     = new JCheckBox("Fast-Forward (skip when only elites remain)", true);
        JCheckBox useNEATCheck         = new JCheckBox("Use NEAT (evolve topology automatically)");

        // Grey out fixed-topology-only fields when NEAT is chosen
        useNEATCheck.addActionListener(ev -> {
            boolean neat = useNEATCheck.isSelected();
            topologyField.setEnabled(!neat);
            elitismField.setEnabled(!neat);
            tourneyField.setEnabled(!neat);
            mutationField.setEnabled(!neat);
        });

        // ── Column 1: Fixed-Topology GA settings ────────────────────────────
        JPanel col1 = new JPanel(new GridBagLayout());
        col1.setBorder(javax.swing.BorderFactory.createTitledBorder(
                javax.swing.BorderFactory.createLineBorder(new Color(150,170,200), 1),
                "Fixed Topology (ignored in NEAT mode)",
                javax.swing.border.TitledBorder.LEFT,
                javax.swing.border.TitledBorder.TOP,
                new Font("SansSerif", Font.BOLD, 11), new Color(60,80,120)));
        addRow(col1, "Hidden Layers (e.g. 6,4,6):",  topologyField,     0);
        addRow(col1, "Elitism (%):",                   elitismField,      1);
        addRow(col1, "Mutation Rate (%):",             mutationField,     2);
        addRow(col1, "Mutation Magnitude:",            mutMagField,       3);
        addRow(col1, "Tournament Size:",               tourneyField,      4);

        // ── Column 2: General simulation settings ───────────────────────────
        JPanel col2 = new JPanel(new GridBagLayout());
        col2.setBorder(javax.swing.BorderFactory.createTitledBorder(
                javax.swing.BorderFactory.createLineBorder(new Color(150,170,200), 1),
                "General Settings",
                javax.swing.border.TitledBorder.LEFT,
                javax.swing.border.TitledBorder.TOP,
                new Font("SansSerif", Font.BOLD, 11), new Color(60,80,120)));
        addRow(col2, "Generations to Train:",         gensField,          0);
        addRow(col2, "Max Frames per Gen:",            maxFramesField,     1);
        addRow(col2, "Population Size:",               popField,           2);
        addRow(col2, "Ghost Render (%):",              ghostPercentField,  3);
        addRow(col2, "Starting Angle (°):",            angleField,         4);
        addRow(col2, "Angle Variance (±°):",           varianceField,      5);
        addRow(col2, "Alt Fitness – Down Time (s):",   maxDownTimeField,   6);
        addRow(col2, "Alt Fitness – Down Y (−1–1):",   downThresholdField, 7);

        // ── Column 3: Toggles + evolution mode ──────────────────────────────
        JPanel col3 = new JPanel();
        col3.setLayout(new BoxLayout(col3, BoxLayout.Y_AXIS));
        col3.setBorder(javax.swing.BorderFactory.createTitledBorder(
                javax.swing.BorderFactory.createLineBorder(new Color(150,170,200), 1),
                "Options & Evolution Mode",
                javax.swing.border.TitledBorder.LEFT,
                javax.swing.border.TitledBorder.TOP,
                new Font("SansSerif", Font.BOLD, 11), new Color(60,80,120)));
        col3.add(polePanel);
        col3.add(Box.createVerticalStrut(6));
        altFitnessCheck     .setAlignmentX(Component.LEFT_ALIGNMENT);
        useCenterRewardCheck.setAlignmentX(Component.LEFT_ALIGNMENT);
        fastForwardCheck    .setAlignmentX(Component.LEFT_ALIGNMENT);
        useNEATCheck        .setAlignmentX(Component.LEFT_ALIGNMENT);
        col3.add(altFitnessCheck);
        col3.add(Box.createVerticalStrut(4));
        col3.add(useCenterRewardCheck);
        col3.add(Box.createVerticalStrut(4));
        col3.add(fastForwardCheck);
        col3.add(Box.createVerticalStrut(12));
        JSeparator sep = new JSeparator();
        sep.setAlignmentX(Component.LEFT_ALIGNMENT);
        col3.add(sep);
        col3.add(Box.createVerticalStrut(8));
        JLabel neatHeader = new JLabel("Evolution Mode:");
        neatHeader.setFont(new Font("SansSerif", Font.BOLD, 11));
        neatHeader.setAlignmentX(Component.LEFT_ALIGNMENT);
        col3.add(neatHeader);
        col3.add(Box.createVerticalStrut(4));
        col3.add(useNEATCheck);

        // ── Assemble three columns side by side ─────────────────────────────
        JPanel mainPanel = new JPanel(new GridLayout(1, 3, 12, 0));
        mainPanel.add(col1);
        mainPanel.add(col2);
        mainPanel.add(col3);

        int option = JOptionPane.showConfirmDialog(
                null, mainPanel,
                "Configure AI Brain & Training",
                JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);

        if (option == JOptionPane.OK_OPTION) {
            try {
                useNEAT = useNEATCheck.isSelected();

                if (!useNEAT) {
                    String[] parts = topologyField.getText().split(",");
                    ArrayList<Integer> sizes = new ArrayList<>();
                    for (String p : parts) {
                        String trimmed = p.trim();
                        if (!trimmed.isEmpty()) sizes.add(Math.max(1, Integer.parseInt(trimmed)));
                    }
                    if (!sizes.isEmpty()) {
                        hiddenLayerSizes = sizes.stream().mapToInt(i -> i).toArray();
                        hiddenLayerStr = topologyField.getText();
                    } else {
                        hiddenLayerSizes = new int[]{6};
                        hiddenLayerStr = "6";
                    }
                }

                targetGenerations  = Math.max(1,   Integer.parseInt(gensField.getText()));
                maxFramesPerGen    = Math.max(60,   Integer.parseInt(maxFramesField.getText()));
                populationSize     = Math.max(10,   Integer.parseInt(popField.getText()));
                elitismPercent     = Math.max(0, Math.min(100, Double.parseDouble(elitismField.getText())));
                mutationRate       = Math.max(0, Math.min(100, Double.parseDouble(mutationField.getText())));
                mutationMagnitude  = Math.max(0.01, Double.parseDouble(mutMagField.getText()));
                ghostRenderPercent = Math.max(0, Math.min(100, Double.parseDouble(ghostPercentField.getText())));
                tournamentSize     = Math.max(1,   Integer.parseInt(tourneyField.getText()));
                startingAngleDeg   = Double.parseDouble(angleField.getText());
                angleVarianceDeg   = Math.max(0,  Double.parseDouble(varianceField.getText()));
                pendulumMode       = triplePole.isSelected() ? 3 : doublePole.isSelected() ? 2 : 1;
                useAlternativeFitness = altFitnessCheck.isSelected();
                useCenterReward    = useCenterRewardCheck.isSelected();
                fastForwardChildren = fastForwardCheck.isSelected();
                Config.maxDownTime     = Math.max(0.1,  Double.parseDouble(maxDownTimeField.getText()));
                Config.downYThreshold  = Math.max(-1.0, Math.min(1.0, Double.parseDouble(downThresholdField.getText())));

                if (useNEAT) promptForNEATSettings();

            } catch (NumberFormatException e) {
                System.out.println("Invalid input, using default settings.");
                hiddenLayerSizes = new int[]{6};
            }
        }
    }


    /**
     * Adds a label + text field row to a GridBagLayout panel.
     */
    private void addRow(JPanel panel, String labelText, JTextField field, int row) {
        GridBagConstraints lc = new GridBagConstraints();
        lc.gridx = 0; lc.gridy = row;
        lc.anchor = GridBagConstraints.WEST;
        lc.insets = new Insets(3, 6, 3, 6);
        lc.fill = GridBagConstraints.NONE;
        panel.add(new JLabel(labelText), lc);

        GridBagConstraints fc = new GridBagConstraints();
        fc.gridx = 1; fc.gridy = row;
        fc.anchor = GridBagConstraints.WEST;
        fc.fill = GridBagConstraints.HORIZONTAL;
        fc.weightx = 1.0;
        fc.insets = new Insets(3, 0, 3, 6);
        panel.add(field, fc);
    }

    private void promptForNEATSettings() {
        JTextField compatField     = new JTextField(String.valueOf(neatCompatThreshold), 8);
        JTextField addNodeField    = new JTextField(String.valueOf((int)(neatAddNodeRate * 100)), 8);
        JTextField addConnField    = new JTextField(String.valueOf((int)(neatAddConnRate * 100)), 8);
        JTextField weightRateField = new JTextField(String.valueOf((int)(neatWeightMutRate * 100)), 8);
        JTextField mutMagNeat      = new JTextField(String.valueOf(mutationMagnitude), 8);
        JTextField crossoverField  = new JTextField(String.valueOf((int)(neatCrossoverRate * 100)), 8);
        JTextField survivalField   = new JTextField(String.valueOf((int)(neatSurvivalFraction * 100)), 8);
        JTextField stagnateField   = new JTextField(String.valueOf(neatStagnationLimit), 8);

        // Two-column grid: left column = first 4 params, right = last 4
        JPanel left  = new JPanel(new GridBagLayout());
        JPanel right = new JPanel(new GridBagLayout());

        left.setBorder(javax.swing.BorderFactory.createTitledBorder("Topology & Speciation"));
        addRow(left,  "Compat. Threshold (δ):",        compatField,     0);
        addRow(left,  "Add-Node Rate (%):",             addNodeField,    1);
        addRow(left,  "Add-Connection Rate (%):",       addConnField,    2);
        addRow(left,  "Stagnation Limit (gens):",       stagnateField,   3);

        right.setBorder(javax.swing.BorderFactory.createTitledBorder("Weights & Breeding"));
        addRow(right, "Weight Mutation Rate (%):",      weightRateField, 0);
        addRow(right, "Weight Mutation Magnitude:",     mutMagNeat,      1);
        addRow(right, "Crossover Rate (%):",            crossoverField,  2);
        addRow(right, "Survival Fraction (%):",         survivalField,   3);

        JPanel neatPanel = new JPanel(new GridLayout(1, 2, 12, 0));
        neatPanel.add(left);
        neatPanel.add(right);

        int r = JOptionPane.showConfirmDialog(
                null, neatPanel,
                "NEAT Hyperparameters",
                JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);

        if (r == JOptionPane.OK_OPTION) {
            try {
                neatCompatThreshold  = Double.parseDouble(compatField.getText());
                neatAddNodeRate      = Math.max(0, Math.min(1, Integer.parseInt(addNodeField.getText()) / 100.0));
                neatAddConnRate      = Math.max(0, Math.min(1, Integer.parseInt(addConnField.getText()) / 100.0));
                neatWeightMutRate    = Math.max(0, Math.min(1, Integer.parseInt(weightRateField.getText()) / 100.0));
                mutationMagnitude    = Math.max(0.01, Double.parseDouble(mutMagNeat.getText()));
                neatCrossoverRate    = Math.max(0, Math.min(1, Integer.parseInt(crossoverField.getText()) / 100.0));
                neatSurvivalFraction = Math.max(0.01, Math.min(1.0, Integer.parseInt(survivalField.getText()) / 100.0));
                neatStagnationLimit  = Math.max(1, Integer.parseInt(stagnateField.getText()));
            } catch (NumberFormatException ex) {
                System.out.println("Invalid NEAT input, using defaults.");
            }
        }
    }


    private void initializePopulation() {
        population = new ArrayList<>();
        Random rand = new Random();
        currentGenAngleDeg = startingAngleDeg + (rand.nextDouble() * 2 - 1) * angleVarianceDeg;

        if (useNEAT) {
            // NEAT mode: minimal genomes (inputs directly connected to output, no hidden)
            int inputCount = 4 + (pendulumMode - 1) * 2; // 4, 6, or 8 inputs
            neatTracker = new NEATInnovationTracker(inputCount + 1 + 1);
            neatSpecies.clear();
            for (int i = 0; i < populationSize; i++) {
                NEATGenome genome = new NEATGenome(inputCount, 1, neatTracker);
                genome.mutateWeights(1.0, 1.0);
                population.add(new CartAgent(genome, this));
            }
        } else {
            // Fixed-topology mode
            int inputCount = 4 + (pendulumMode - 1) * 2; // 4, 6, or 8 inputs
            int[] topology = new int[hiddenLayerSizes.length + 2];
            topology[0] = inputCount;
            for (int i = 0; i < hiddenLayerSizes.length; i++) {
                topology[i + 1] = hiddenLayerSizes[i];
            }
            topology[topology.length - 1] = 1;
            for (int i = 0; i < populationSize; i++) {
                population.add(new CartAgent(new NeuralNetwork(topology), this));
            }
        }

        previousGenerationFitness = null;
        performanceHistory.clear();
    }


    private void updatePhysics() {
        boolean allDead = true;
        boolean allChildrenDead = true;
        int childrenCount = 0;
        CartAgent bestCurrentAgent = population.get(0);

        for (CartAgent agent : population) {
            agent.update();
            if (agent.alive) {
                allDead = false;
                if (!agent.isElite) {
                    allChildrenDead = false;
                }
            }
            if (!agent.isElite) {
                childrenCount++;
            }
            if (agent.fitness > bestCurrentAgent.fitness) {
                bestCurrentAgent = agent;
            }
            // Track the absolute best brain ever achieved across all generations
            if (agent.fitness > bestFitnessEver) {
                bestFitnessEver = agent.fitness;
                globalBestBrain = agent.brain.copy();
            }
        }
        if (!renderCheckBox.isSelected()) {
            updateCamera();
        }

        framesThisGen++;

        boolean skipGeneration = fastForwardChildren && childrenCount > 0 && allChildrenDead;

        // End of generation if all carts failed, time limit reached, or only elites are left
        if (allDead || framesThisGen >= maxFramesPerGen || skipGeneration) {
            if (generation >= targetGenerations) {
                captureGenerationMetrics(); // Capture final generation metrics
                calculateFinalStats();
                simState = SimulationState.WAITING_FOR_REPLAY;
                startReplayBtn.setVisible(true);
                testNewAngleBtn.setVisible(true);
                continueTrainingBtn.setVisible(true);
            } else {
                evolveNextGeneration();
            }
        }
    }

    private void calculateFinalStats() {
        if (population == null || population.isEmpty()) return;
        double sum = 0;
        double[] fitnesses = new double[population.size()];
        for (int i = 0; i < population.size(); i++) {
            double f = population.get(i).fitness;
            fitnesses[i] = f;
            sum += f;
        }
        Arrays.sort(fitnesses);
        finalBestFitness = fitnesses[fitnesses.length - 1];
        finalMeanFitness = sum / population.size();
        finalMedianFitness = fitnesses[fitnesses.length / 2];
        
        // Calculate Standard Deviation
        double varianceSum = 0;
        for (int i = 0; i < population.size(); i++) {
            varianceSum += Math.pow(fitnesses[i] - finalMeanFitness, 2);
        }
        finalStdDevFitness = Math.sqrt(varianceSum / population.size());
    }

    private void startReplay() {
        currentGenAngleDeg = startingAngleDeg; // Replay runs strictly at the base target angle
        
        // Reset the population physically to act as ghosts alongside the best agent
        for (int i = 0; i < population.size(); i++) {
            population.set(i, new CartAgent(population.get(i).brain, this));
        }
        
        if (globalBestBrain != null) {
            replayAgent = new CartAgent(globalBestBrain, this);
        } else {
            replayAgent = new CartAgent(population.get(0).brain, this); // Fallback
        }
        framesThisGen = 0;
    }

    private void updateReplay() {
        replayAgent.update();
        for (CartAgent agent : population) {
            agent.update();
        }
        if (!renderCheckBox.isSelected()) {
            updateCamera();
        }
        
        framesThisGen++;
        // Loop the replay if the agent dies or runs out of time
        if (!replayAgent.alive || framesThisGen >= maxFramesPerGen) {
            startReplay();
        }
    }

    private void evolveNextGeneration() {
        // Capture metrics before breeding
        captureGenerationMetrics();

        // Save the fitness data before we breed the new generation!
        previousGenerationFitness = new double[population.size()];
        for (int i = 0; i < population.size(); i++) {
            previousGenerationFitness[i] = population.get(i).fitness;
        }

        // Calculate new base angle for the upcoming generation
        Random rand = new Random();
        currentGenAngleDeg = startingAngleDeg + (rand.nextDouble() * 2 - 1) * angleVarianceDeg;

        if (useNEAT) {
            evolveNEAT(rand);
        } else {
            evolveFixedTopology(rand);
        }

        generation++;
        framesThisGen = 0;
    }

    /** Original fixed-topology GA evolution (elitism + tournament crossover). */
    private void evolveFixedTopology(Random rand) {
        // Sort population by fitness descending
        population.sort((a, b) -> Double.compare(b.fitness, a.fitness));

        ArrayList<CartAgent> newPopulation = new ArrayList<>();

        // Elitism: Keep the top X% best performers exactly as they are
        int eliteCount = (int) (populationSize * (elitismPercent / 100.0));
        for (int i = 0; i < eliteCount && i < population.size(); i++) {
            CartAgent elite = new CartAgent(population.get(i).brain.copy(), this);
            elite.isElite = true;
            newPopulation.add(elite);
        }

        // Breed the rest
        while (newPopulation.size() < populationSize) {
            CartAgent parentA = tournamentSelection(rand);
            CartAgent parentB = tournamentSelection(rand);
            NeuralNetwork childBrain = ((NeuralNetwork) parentA.brain)
                    .breed((NeuralNetwork) parentB.brain, mutationRate, mutationMagnitude);
            newPopulation.add(new CartAgent(childBrain, this));
        }

        population = newPopulation;
    }

    /** NEAT speciation-based evolution. */
    private void evolveNEAT(Random rand) {
        // Reset innovation tracker cache for this generation
        neatTracker.resetGenerationCache();

        // ── 1. Assign genomes to species ──────────────────────────────────────
        // Keep old representatives; clear member lists
        for (NEATSpecies sp : neatSpecies) sp.clearMembers();

        for (CartAgent agent : population) {
            NEATGenome genome = (NEATGenome) agent.brain;
            boolean placed = false;
            for (NEATSpecies sp : neatSpecies) {
                if (genome.compatibilityDistance(sp.representative, 1.0, 1.0, 0.4) < neatCompatThreshold) {
                    sp.add(genome);
                    sp.fitnesses.add(agent.fitness);
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                NEATSpecies newSp = new NEATSpecies(genome);
                newSp.add(genome);
                newSp.fitnesses.add(agent.fitness);
                neatSpecies.add(newSp);
            }
        }

        // Remove empty species (may be left over from previous generation)
        neatSpecies.removeIf(sp -> sp.members.isEmpty());

        // ── 2. Stagnation check ───────────────────────────────────────────────
        neatSpecies.removeIf(sp -> sp.updateStagnation(neatStagnationLimit) && neatSpecies.size() > 1);

        // ── 3. Compute total adjusted fitness for offspring allocation ─────────
        double totalAdjustedFitness = 0;
        for (NEATSpecies sp : neatSpecies) {
            totalAdjustedFitness += sp.totalAdjustedFitness();
        }

        // ── 4. Cull bottom performers within each species ─────────────────────
        for (NEATSpecies sp : neatSpecies) sp.cull(neatSurvivalFraction);

        // ── 5. Breed offspring ────────────────────────────────────────────────
        ArrayList<CartAgent> newPopulation = new ArrayList<>();

        for (NEATSpecies sp : neatSpecies) {
            // Allocate offspring proportionally to adjusted fitness
            int offspring = (int) Math.round(
                    (sp.totalAdjustedFitness() / Math.max(1e-9, totalAdjustedFitness)) * populationSize);
            offspring = Math.max(1, offspring); // every species gets at least 1 slot

            // Champion carry-over: preserve best genome unchanged if species is large enough
            if (sp.members.size() >= 5) {
                NEATGenome champion = sp.selectParent(rand);
                newPopulation.add(new CartAgent(champion.copy(), this));
                offspring--;
            }

            for (int i = 0; i < offspring && newPopulation.size() < populationSize; i++) {
                NEATGenome child;
                if (sp.members.size() > 1 && rand.nextDouble() < neatCrossoverRate) {
                    NEATGenome parentA = sp.selectParent(rand);
                    NEATGenome parentB = sp.selectParent(rand);
                    // parentA is the "fitter" by convention (it's fine either way)
                    child = parentA.crossover(parentB);
                } else {
                    child = (NEATGenome) sp.selectParent(rand).copy();
                }

                // Apply mutations
                child.mutateWeights(neatWeightMutRate, mutationMagnitude);
                if (rand.nextDouble() < neatAddNodeRate) child.mutateAddNode(neatTracker);
                if (rand.nextDouble() < neatAddConnRate) child.mutateAddConnection(neatTracker);

                newPopulation.add(new CartAgent(child, this));
            }

            // Update species representative for next generation
            sp.updateRepresentative(rand);
        }

        // Fill any remaining slots with mutations of random surviving genomes
        while (newPopulation.size() < populationSize) {
            NEATSpecies sp = neatSpecies.get(rand.nextInt(neatSpecies.size()));
            if (sp.members.isEmpty()) continue;
            NEATGenome child = (NEATGenome) sp.selectParent(rand).copy();
            child.mutateWeights(neatWeightMutRate, mutationMagnitude);
            if (rand.nextDouble() < neatAddConnRate) child.mutateAddConnection(neatTracker);
            newPopulation.add(new CartAgent(child, this));
        }

        population = newPopulation;
    }


    private void captureGenerationMetrics() {
        if (population == null || population.isEmpty()) return;
        double sum = 0;
        double[] fitnesses = new double[population.size()];
        for (int i = 0; i < population.size(); i++) {
            double f = population.get(i).fitness;
            fitnesses[i] = f;
            sum += f;
        }
        Arrays.sort(fitnesses);
        double best = fitnesses[fitnesses.length - 1];
        double mean = sum / population.size();
        double median = fitnesses[fitnesses.length / 2];
        double q1 = fitnesses[fitnesses.length / 4];
        double q3 = fitnesses[fitnesses.length * 3 / 4];
        
        synchronized(performanceHistory) {
            performanceHistory.add(new GenerationData(generation, best, median, mean, q1, q3));
        }

        if (liveGraphFrame != null && liveGraphFrame.isVisible()) {
            liveGraphFrame.repaint();
            FitnessHistogram hist = (FitnessHistogram) liveGraphFrame.getRootPane().getClientProperty("histogramPanel");
            if (hist != null) {
                hist.updateData(previousGenerationFitness);
            }
        }
    }

    private void exportMetricsToCSV() {
        synchronized(performanceHistory) {
            if (performanceHistory.isEmpty()) {
                JOptionPane.showMessageDialog(this, "No metrics to export yet.");
                return;
            }

            JFileChooser chooser = new JFileChooser();
            chooser.setDialogTitle("Export Performance Metrics as CSV");
            if (chooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
                File file = chooser.getSelectedFile();
                if (!file.getName().toLowerCase().endsWith(".csv")) {
                    file = new File(file.getAbsolutePath() + ".csv");
                }

                try (PrintWriter pw = new PrintWriter(new FileWriter(file))) {
                    pw.println("Generation,Best Fitness,Median Fitness,Mean Fitness");
                    for (GenerationData data : performanceHistory) {
                        pw.printf("%d,%.4f,%.4f,%.4f%n", data.generation, data.bestFitness, data.medianFitness, data.meanFitness);
                    }
                    JOptionPane.showMessageDialog(this, "Metrics exported successfully to " + file.getName());
                } catch (IOException ex) {
                    JOptionPane.showMessageDialog(this, "Export failed: " + ex.getMessage());
                }
            }
        }
    }

    private void updateCamera() {
        double minX = 400;
        double maxX = 400;
        boolean anyAlive = false;

        if (simState == SimulationState.REPLAY) {
            if (replayAgent != null && replayAgent.alive) {
                anyAlive = true;
                minX = Math.min(minX, replayAgent.cartX);
                maxX = Math.max(maxX, replayAgent.cartX);
            }
            for (CartAgent agent : population) {
                if (agent.alive) {
                    anyAlive = true;
                    minX = Math.min(minX, agent.cartX);
                    maxX = Math.max(maxX, agent.cartX);
                }
            }
        } else {
            for (CartAgent agent : population) {
                if (agent.alive) {
                    anyAlive = true;
                    minX = Math.min(minX, agent.cartX);
                    maxX = Math.max(maxX, agent.cartX);
                }
            }
        }

        if (anyAlive) {
            double worldWidth = (maxX - minX) + 200; // 100px padding each side
            double targetZoom = Math.min(1.0, (double)getWidth() / worldWidth);
            cameraZoom += (targetZoom - cameraZoom) * 0.05; 
            
            double targetX = (minX + maxX) / 2.0;
            cameraX += (targetX - cameraX) * 0.05;
        } else {
            cameraZoom += (1.0 - cameraZoom) * 0.05;
            cameraX += (400.0 - cameraX) * 0.05;
        }
    }

    private CartAgent tournamentSelection(Random rand) {
        // Pick random agents, return the best among them
        CartAgent best = population.get(rand.nextInt(populationSize));
        for (int i = 0; i < tournamentSize - 1; i++) {
            CartAgent competitor = population.get(rand.nextInt(populationSize));
            if (competitor.fitness > best.fitness) {
                best = competitor;
            }
        }
        return best;
    }
    
    private void exportBrain() {
        Brain brainToSave = globalBestBrain != null ? globalBestBrain : population.get(0).brain;
        if (brainToSave == null) return;

        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Export Best AI Brain");
        if (chooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(chooser.getSelectedFile()))) {
                oos.writeObject(brainToSave);
                JOptionPane.showMessageDialog(this, "AI Brain exported successfully!");
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "Export failed: " + ex.getMessage());
            }
        }
    }

    private void importBrain() {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Import AI Brain");
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(chooser.getSelectedFile()))) {
                Object obj = ois.readObject();
                Brain imported;
                if (obj instanceof NEATGenome) {
                    imported = (NEATGenome) obj;
                    useNEAT = true;
                    NEATGenome ng = (NEATGenome) imported;
                    pendulumMode = 1 + (ng.inputCount - 4) / 2; // 4→1, 6→2, 8→3
                    neatTracker = new NEATInnovationTracker(ng.inputCount + 1 + 1);
                    neatSpecies.clear();
                    population.clear();
                    population.add(new CartAgent(imported.copy(), this));
                    Random rand = new Random();
                    for (int i = 1; i < populationSize; i++) {
                        NEATGenome child = (NEATGenome) imported.copy();
                        child.mutateWeights(neatWeightMutRate, mutationMagnitude);
                        if (rand.nextDouble() < neatAddConnRate) child.mutateAddConnection(neatTracker);
                        population.add(new CartAgent(child, this));
                    }
                } else {
                    imported = (NeuralNetwork) obj;
                    useNEAT = false;
                    NeuralNetwork nn = (NeuralNetwork) imported;
                    int[] top = nn.topology;
                    hiddenLayerSizes = new int[top.length - 2];
                    System.arraycopy(top, 1, hiddenLayerSizes, 0, hiddenLayerSizes.length);
                    pendulumMode = 1 + (top[0] - 4) / 2; // 4→1, 6→2, 8→3
                    population.clear();
                    population.add(new CartAgent(imported.copy(), this));
                    for (int i = 1; i < populationSize; i++) {
                        population.add(new CartAgent(nn.breed(nn, mutationRate, mutationMagnitude), this));
                    }
                }

                globalBestBrain = imported;
                generation = 1;
                framesThisGen = 0;
                bestFitnessEver = 0;
                previousGenerationFitness = null;
                simState = SimulationState.TRAINING;
                startReplayBtn.setVisible(false);
                testNewAngleBtn.setVisible(false);
                continueTrainingBtn.setVisible(false);

                String mode = useNEAT ? "NEAT" : "Fixed-Topology";
                JOptionPane.showMessageDialog(this,
                        "AI Brain imported (%s)! Evolving a new swarm based on this Champion.".formatted(mode));
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "Import failed: " + ex.getMessage());
            }
        }
    }

    private void drawGrid(Graphics2D g2, int originX, int originY) {
        // Calculate the visible world bounds
        double viewWidthWorld = getWidth() / cameraZoom;
        double viewHeightWorld = getHeight() / cameraZoom;
        
        double left = cameraX - viewWidthWorld / 2.0;
        double right = cameraX + viewWidthWorld / 2.0;
        double top = originY - viewHeightWorld / 2.0;
        double bottom = originY + viewHeightWorld / 2.0;

        // Base step size on a fixed world unit (50px at 1x zoom)
        int stepSize = 50; 
        
        // Determine the range of steps to draw
        int startX = (int) Math.floor((left - originX) / stepSize);
        int endX = (int) Math.ceil((right - originX) / stepSize);
        int startY = (int) Math.floor((top - originY) / stepSize);
        int endY = (int) Math.ceil((bottom - originY) / stepSize);

        // Draw the grid lines
        for (int i = startX; i <= endX; i++) {
            if (i == 0) {
                g2.setColor(new Color(100, 100, 100, 255));
                g2.setStroke(new BasicStroke(2.0f / (float)cameraZoom)); // Scale stroke so it stays sharp
            } else if (i % 4 == 0) {
                g2.setColor(new Color(100, 100, 100, 127));
                g2.setStroke(new BasicStroke(1.0f / (float)cameraZoom));
            } else if (i % 2 == 0) {
                g2.setColor(new Color(100, 100, 100, 63));
                g2.setStroke(new BasicStroke(1.0f / (float)cameraZoom));
            } else {
                g2.setColor(new Color(100, 100, 100, 31));
                g2.setStroke(new BasicStroke(1.0f / (float)cameraZoom));
            }
            
            int x = originX + (i * stepSize);
            g2.drawLine(x, (int)top, x, (int)bottom);
        }

        for (int j = startY; j <= endY; j++) {
            if (j == 0) {
                g2.setColor(new Color(100, 100, 100, 255));
                g2.setStroke(new BasicStroke(2.0f / (float)cameraZoom));
            } else if (j % 4 == 0) {
                g2.setColor(new Color(100, 100, 100, 127));
                g2.setStroke(new BasicStroke(1.0f / (float)cameraZoom));
            } else if (j % 2 == 0) {
                g2.setColor(new Color(100, 100, 100, 63));
                g2.setStroke(new BasicStroke(1.0f / (float)cameraZoom));
            } else {
                g2.setColor(new Color(100, 100, 100, 31));
                g2.setStroke(new BasicStroke(1.0f / (float)cameraZoom));
            }
            
            int y = originY + (j * stepSize);
            g2.drawLine((int)left, y, (int)right, y);
        }
        
        g2.setStroke(new BasicStroke(1.0f));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int centerY = getHeight() / 2 + 100;

        // 1. Apply World Transformation (Zoom/Pan)
        java.awt.geom.AffineTransform screenTransform = g2.getTransform();
        g2.translate(getWidth() / 2.0, centerY);
        g2.scale(cameraZoom, cameraZoom);
        g2.translate(-cameraX, -centerY);

        // Draw the coordinate grid
        drawGrid(g2, 400, centerY);

        CartAgent bestAgent = null;
        if (simState == SimulationState.TRAINING || simState == SimulationState.WAITING_FOR_REPLAY) {
            if (!renderCheckBox.isSelected()) {
                // Find the best current agent to highlight
                for (CartAgent agent : population) {
                    if (agent.alive) {
                        if (bestAgent == null || agent.fitness > bestAgent.fitness) {
                            bestAgent = agent;
                        }
                    }
                }

                // Draw Ghost Agents
                g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.25f));
                int renderLimit = (int) (populationSize * (ghostRenderPercent / 100.0));
                int renderedCount = 0;
                for (CartAgent agent : population) {
                    if (agent.alive && agent != bestAgent) {
                        if (renderedCount < renderLimit) {
                            drawAgent(g2, agent, centerY, Color.LIGHT_GRAY, new Color(150, 150, 150));
                            renderedCount++;
                        }
                    }
                }

                // Draw Best Agent
                if (bestAgent != null) {
                    g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.75f));
                    drawAgent(g2, bestAgent, centerY, new Color(41, 128, 185), new Color(139, 69, 19));
                }
            }
        } else if (simState == SimulationState.REPLAY) {
            // Replay Mode Drawing
            g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.25f));
            int renderLimit = (int) (populationSize * (ghostRenderPercent / 100.0));
            int renderedCount = 0;
            for (CartAgent agent : population) {
                if (agent.alive) {
                    if (renderedCount < renderLimit) {
                        drawAgent(g2, agent, centerY, Color.LIGHT_GRAY, new Color(150, 150, 150));
                        renderedCount++;
                    }
                }
            }
            
            g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
            if (replayAgent != null) {
                drawAgent(g2, replayAgent, centerY, new Color(41, 128, 185), new Color(139, 69, 19));
            }
        }

        // 2. Reset Transform for UI elements (Text, NN, Histogram)
        g2.setTransform(screenTransform);
        g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));

        if (simState == SimulationState.TRAINING || simState == SimulationState.WAITING_FOR_REPLAY) {
            if (renderCheckBox.isSelected()) {
                g2.setColor(new Color(231, 76, 60));
                g2.setFont(new Font("SansSerif", Font.BOLD, 20));
                g2.drawString("SUPER FAST MODE ACTIVE - BYPASSING RENDER", 20, getHeight() - 50);
            }

            g2.setColor(Color.DARK_GRAY);
            g2.setFont(new Font("SansSerif", Font.BOLD, 16));
            g2.drawString("TRAINING - Generation: " + generation + " / " + targetGenerations, 20, 30);
            g2.drawString(String.format("Frame: %d / %d | Speed: %dx", framesThisGen, maxFramesPerGen, simSpeedSlider.getValue()), 20, 50);
            g2.drawString(String.format("Gen Angle: %.1f\u00B0", currentGenAngleDeg), 20, 70);
            
            if (!renderCheckBox.isSelected()) {
                int aliveCount = (int) population.stream().filter(a -> a.alive).count();
                g2.drawString("Alive: " + aliveCount + " / " + populationSize, 20, 90);
            }
            g2.drawString(String.format("Best Fitness Ever: %.1f", bestFitnessEver), 20, 110);
            if (useNEAT) {
                g2.drawString("Species: " + neatSpecies.size() + " | Nodes: " +
                        (bestAgent != null ? ((NEATGenome)bestAgent.brain).nodes.size() : "?") +
                        " | Conns: " +
                        (bestAgent != null ? ((NEATGenome)bestAgent.brain).connections.size() : "?"), 20, 130);
                g2.drawString("NN Visualizer [V]: " + (showNN ? "On" : "Off"), 20, 150);
            } else {
                g2.drawString("Histogram [H]: " + (showHistogram ? "On" : "Off"), 20, 130);
                g2.drawString("NN Visualizer [V]: " + (showNN ? "On" : "Off"), 20, 150);
            }

            if (showHistogram && !renderCheckBox.isSelected()) {
                drawFitnessHistogram(g2, 20, getHeight() - 170, 300, 150);
            }

            int nnWidth = Math.max(200, getWidth() / 4);
            int nnHeight = Math.max(150, getHeight() / 3);
            int nnStartX = getWidth() - nnWidth - 80;
            int nnStartY = 50;

            if (showNN && bestAgent != null) {
                if (useNEAT) {
                    drawNEATGenome(g2, (NEATGenome) bestAgent.brain, nnStartX, nnStartY, nnWidth, nnHeight);
                } else {
                    drawNeuralNetwork(g2, bestAgent.brain, nnStartX, nnStartY, nnWidth, nnHeight);
                }
            }
            
            if (simState == SimulationState.WAITING_FOR_REPLAY) {
                g2.setColor(Color.RED);
                g2.drawString("Training Complete! Choose an action below to proceed.", 20, 180);
                g2.setColor(Color.DARK_GRAY);
                g2.drawString(String.format("Final Gen -> Best: %.1f | Mean: %.1f | Median: %.1f | StdDev: %.1f", 
                                finalBestFitness, finalMeanFitness, finalMedianFitness, finalStdDevFitness), 20, 200);
            }
        } else if (simState == SimulationState.REPLAY) {
            g2.setColor(Color.DARK_GRAY);
            g2.setFont(new Font("SansSerif", Font.BOLD, 16));
            g2.drawString("REPLAY MODE - Best Agent Overall", 20, 30);
            g2.drawString(String.format("Frame: %d / %d | Speed: %dx", framesThisGen, maxFramesPerGen, simSpeedSlider.getValue()), 20, 50);
            g2.drawString(String.format("Replay Angle: %.1f\u00B0", currentGenAngleDeg), 20, 70);
            g2.drawString(String.format("Best Fitness Achieved: %.1f", bestFitnessEver), 20, 90);
            g2.drawString(String.format("Final Gen -> Best: %.1f | Mean: %.1f | Median: %.1f | StdDev: %.1f", 
                            finalBestFitness, finalMeanFitness, finalMedianFitness, finalStdDevFitness), 20, 110);
            g2.drawString("Histogram [H]: " + (showHistogram ? "On" : "Off"), 20, 130);
            g2.drawString("NN Visualizer [V]: " + (showNN ? "On" : "Off"), 20, 150);

            if (showHistogram) {
                drawFitnessHistogram(g2, 20, getHeight() - 170, 300, 150);
            }
            
            int nnWidth = Math.max(200, getWidth() / 4);
            int nnHeight = Math.max(150, getHeight() / 3);
            int nnStartX = getWidth() - nnWidth - 80;
            int nnStartY = 50;

            if (showNN && replayAgent != null) {
                if (useNEAT) {
                    drawNEATGenome(g2, (NEATGenome) replayAgent.brain, nnStartX, nnStartY, nnWidth, nnHeight);
                } else {
                    drawNeuralNetwork(g2, replayAgent.brain, nnStartX, nnStartY, nnWidth, nnHeight);
                }
            }
            
            if (replayAgent != null && !replayAgent.alive) {
                g2.setColor(Color.RED);
                g2.drawString("Agent Failed! Resetting...", 20, 180);
            }
        }
    }

    private void drawAgent(Graphics2D g2, CartAgent agent, int centerY, Color cartColor, Color poleColor) {
        int centerX = (int) agent.cartX;

        // Draw First Pole
        int poleEndX = (int) (centerX + Math.sin(agent.poleAngle) * Config.POLE_LENGTH_PX);
        int poleEndY = (int) (centerY + Math.cos(agent.poleAngle) * Config.POLE_LENGTH_PX);

        g2.setStroke(new BasicStroke(6, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        g2.setColor(poleColor);
        g2.drawLine(centerX, centerY, poleEndX, poleEndY);

        // Draw Second Pole if applicable
        if (pendulumMode >= 2) {
            int pole2EndX = (int) (poleEndX + Math.sin(agent.poleAngle2) * Config.POLE_LENGTH_PX);
            int pole2EndY = (int) (poleEndY + Math.cos(agent.poleAngle2) * Config.POLE_LENGTH_PX);

            g2.drawLine(poleEndX, poleEndY, pole2EndX, pole2EndY);

            // Joint 1 dot (orange)
            g2.setColor(new Color(230, 100, 20));
            g2.fillOval(poleEndX - 5, poleEndY - 5, 10, 10);

            // Draw Third Pole if applicable
            if (pendulumMode == 3) {
                int pole3EndX = (int) (pole2EndX + Math.sin(agent.poleAngle3) * Config.POLE_LENGTH_PX);
                int pole3EndY = (int) (pole2EndY + Math.cos(agent.poleAngle3) * Config.POLE_LENGTH_PX);

                g2.setColor(poleColor);
                g2.drawLine(pole2EndX, pole2EndY, pole3EndX, pole3EndY);

                // Joint 2 dot (purple)
                g2.setColor(new Color(150, 50, 200));
                g2.fillOval(pole2EndX - 5, pole2EndY - 5, 10, 10);
            }
        }

        // Draw Cart
        g2.setColor(cartColor);
        g2.fillRect(centerX - Config.CART_WIDTH / 2, centerY - Config.CART_HEIGHT / 2, Config.CART_WIDTH, Config.CART_HEIGHT);
        
        // Draw Wheels
        g2.setColor(Color.BLACK);
        g2.fillOval(centerX - Config.CART_WIDTH / 2 + 5, centerY + Config.CART_HEIGHT / 2 - 10, 20, 20);
        g2.fillOval(centerX + Config.CART_WIDTH / 2 - 25, centerY + Config.CART_HEIGHT / 2 - 10, 20, 20);

        // Draw Base Joint
        g2.setColor(Color.RED);
        g2.fillOval(centerX - 5, centerY - 5, 10, 10);
    }

    /**
     * Renders a live visual of the Neural Network and its activating nodes.
     */
    private void drawNeuralNetwork(Graphics2D g2, Brain brainArg, int startX, int startY, int width, int height) {
        NeuralNetwork brain = (NeuralNetwork) brainArg;
        if (brain == null || brain.lastActivations == null) return;
        
        int[] top = brain.topology;
        int numLayers = top.length;
        double layerSpacing = numLayers > 1 ? (double) width / (numLayers - 1) : 0;
        
        g2.setColor(Color.DARK_GRAY);
        g2.setFont(new Font("SansSerif", Font.BOLD, 12));
        g2.drawString("Live Neural Network", startX, startY - 15);
        
        // Draw connections (faint lines between nodes)
        for (int i = 0; i < numLayers - 1; i++) {
            int nodesCurrent = top[i];
            int nodesNext = top[i+1];
            
            double x1 = startX + i * layerSpacing;
            double x2 = startX + (i + 1) * layerSpacing;
            
            double spacing1 = (double) height / Math.max(1, nodesCurrent);
            double spacing2 = (double) height / Math.max(1, nodesNext);
            
            for (int j = 0; j < nodesCurrent; j++) {
                double y1 = startY + j * spacing1 + spacing1 / 2;
                for (int k = 0; k < nodesNext; k++) {
                    double y2 = startY + k * spacing2 + spacing2 / 2;
                    
                    // Retrieve the weight multiplier for this connection
                    // Array structure is: [layer][targetNode][sourceNode]
                    double weight = brain.weights[i][k][j];
                    
                    // Map weight magnitude to line thickness (using user slider scale)
                    float userScale = thicknessSlider.getValue() / 10.0f;
                    float thicknessScale = userScale * (float) Math.max(1.0, height / 100.0);
                    float thickness = (float) Math.max(0.2, Math.min(height / 5.0, Math.abs(weight) * thicknessScale));
                    g2.setStroke(new BasicStroke(thickness, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
                    
                    // Color code the multiplier: Blue for positive, Red for negative
                    if (weight > 0) {
                        g2.setColor(new Color(41, 128, 185, 120)); // Transparent Blue
                    } else {
                        g2.setColor(new Color(192, 57, 43, 120)); // Transparent Red
                    }
                    
                    g2.drawLine((int)x1, (int)y1, (int)x2, (int)y2);
                }
            }
        }
        
        // Reset stroke before drawing the circular nodes
        g2.setStroke(new BasicStroke(1));
        
        // Dynamically scale node radius based on panel height and max nodes
        int maxNodes = 1;
        for (int n : top) {
            maxNodes = Math.max(maxNodes, n);
        }
        int nodeRadius = Math.max(6, Math.min(25, height / (maxNodes * 3)));
        
        // Draw nodes (Grayscale based on activation value)
        for (int i = 0; i < numLayers; i++) {
            int nodes = top[i];
            double x = startX + i * layerSpacing;
            double spacing = (double) height / Math.max(1, nodes);
            
            for (int j = 0; j < nodes; j++) {
                double y = startY + j * spacing + spacing / 2;
                
                double val = brain.lastActivations[i][j];
                
                // Map the activation val [-1, 1] to a grayscale intensity [0, 255]
                // -1 -> 0 (Black), 0 -> 127 (Gray), 1 -> 255 (White)
                int intensity = (int) ((Math.max(-1.0, Math.min(1.0, val)) + 1.0) / 2.0 * 255);
                
                g2.setColor(new Color(intensity, intensity, intensity));
                g2.fillOval((int)x - nodeRadius, (int)y - nodeRadius, nodeRadius * 2, nodeRadius * 2);
                
                g2.setColor(Color.BLACK);
                g2.drawOval((int)x - nodeRadius, (int)y - nodeRadius, nodeRadius * 2, nodeRadius * 2);
                
                // Add labels for the Input Layer
                if (i == 0) {
                    String[] labels = pendulumMode == 3
                        ? new String[]{"Cart X", "Cart Vel", "Angle 1", "Ang Vel 1", "Angle 2", "Ang Vel 2", "Angle 3", "Ang Vel 3"}
                        : pendulumMode == 2
                        ? new String[]{"Cart X", "Cart Vel", "Angle 1", "Ang Vel 1", "Angle 2", "Ang Vel 2"}
                        : new String[]{"Cart X", "Cart Vel", "Pole Angle", "Pole Vel"};
                        
                    if (j < labels.length) {
                        g2.setColor(Color.DARK_GRAY);
                        g2.setFont(new Font("SansSerif", Font.BOLD, 10));
                        FontMetrics fm = g2.getFontMetrics();
                        int textWidth = fm.stringWidth(labels[j]);
                        // Draw label to the left of the node
                        g2.drawString(labels[j], (int)x - nodeRadius - textWidth - 6, (int)y + 4);
                    }
                }
                
                // Add label for the Output Layer
                if (i == numLayers - 1) {
                    g2.setColor(Color.DARK_GRAY);
                    g2.setFont(new Font("SansSerif", Font.BOLD, 11));
                    // Draw label to the right of the node
                    g2.drawString("Force", (int)x + nodeRadius + 6, (int)y + 4);
                }
            }
        }
    }

    /**
     * Renders a NEAT genome as an arbitrary graph.
     * Nodes are positioned by type: inputs on the left, outputs on the right,
     * hidden nodes in a grid in the middle. Color-coded connections show weights.
     */
    private void drawNEATGenome(Graphics2D g2, NEATGenome genome, int startX, int startY, int width, int height) {
        if (genome == null) return;

        g2.setColor(Color.DARK_GRAY);
        g2.setFont(new Font("SansSerif", Font.BOLD, 12));
        g2.drawString("Live NEAT Network", startX, startY - 15);

        // Assign positions to each node
        java.util.Map<Integer, int[]> positions = new HashMap<>();

        List<NEATGenome.NodeGene> inputs   = new ArrayList<>();
        List<NEATGenome.NodeGene> outputs  = new ArrayList<>();
        List<NEATGenome.NodeGene> hidden   = new ArrayList<>();
        NEATGenome.NodeGene biasNode = null;

        for (NEATGenome.NodeGene ng : genome.nodes) {
            switch (ng.type) {
                case INPUT:  inputs.add(ng);  break;
                case OUTPUT: outputs.add(ng); break;
                case BIAS:   biasNode = ng;   break;
                default:     hidden.add(ng);  break;
            }
        }

        int nodeR = 10;

        // Inputs: left column
        for (int i = 0; i < inputs.size(); i++) {
            int x = startX + nodeR + 2;
            int y = startY + (int)((i + 0.5) * height / Math.max(1, inputs.size()));
            positions.put(inputs.get(i).id, new int[]{x, y});
        }
        // Bias: below inputs
        if (biasNode != null) {
            int x = startX + nodeR + 2;
            int y = startY + height - nodeR - 2;
            positions.put(biasNode.id, new int[]{x, y});
        }
        // Outputs: right column
        for (int i = 0; i < outputs.size(); i++) {
            int x = startX + width - nodeR - 2;
            int y = startY + (int)((i + 0.5) * height / Math.max(1, outputs.size()));
            positions.put(outputs.get(i).id, new int[]{x, y});
        }
        // Hidden: grid in the middle
        int cols = Math.max(1, (int) Math.ceil(Math.sqrt(hidden.size())));
        int rows = (int) Math.ceil((double) hidden.size() / cols);
        for (int i = 0; i < hidden.size(); i++) {
            int col = i % cols;
            int row = i / cols;
            int x = startX + width / 4 + (int)((col + 0.5) * (width / 2) / Math.max(1, cols));
            int y = startY + (int)((row + 0.5) * height / Math.max(1, rows));
            positions.put(hidden.get(i).id, new int[]{x, y});
        }

        // Draw connections
        float userScale = thicknessSlider.getValue() / 10.0f;
        for (NEATGenome.ConnectionGene cg : genome.connections) {
            int[] from = positions.get(cg.from);
            int[] to   = positions.get(cg.to);
            if (from == null || to == null) continue;

            float thickness = (float) Math.max(0.5, Math.min(6.0, Math.abs(cg.weight) * userScale));
            g2.setStroke(new BasicStroke(thickness, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));

            if (!cg.enabled) {
                g2.setColor(new Color(180, 180, 180, 60));
            } else if (cg.weight > 0) {
                g2.setColor(new Color(41, 128, 185, 150));
            } else {
                g2.setColor(new Color(192, 57, 43, 150));
            }
            g2.drawLine(from[0], from[1], to[0], to[1]);
        }

        g2.setStroke(new BasicStroke(1));

        // Draw nodes
        String[] inputLabels = pendulumMode == 3
                ? new String[]{"Cart X", "Cart V", "Ang 1", "AngV 1", "Ang 2", "AngV 2", "Ang 3", "AngV 3"}
                : pendulumMode == 2
                ? new String[]{"Cart X", "Cart V", "Ang 1", "AngV 1", "Ang 2", "AngV 2"}
                : new String[]{"Cart X", "Cart V", "Angle", "Ang Vel"};

        for (NEATGenome.NodeGene ng : genome.nodes) {
            int[] pos = positions.get(ng.id);
            if (pos == null) continue;

            // Color by type and activation
            Color nodeColor;
            switch (ng.type) {
                case INPUT:
                    int inIntensity = (int)((Math.max(-1, Math.min(1, ng.lastActivation)) + 1) / 2.0 * 255);
                    nodeColor = new Color(inIntensity, inIntensity, 220);
                    break;
                case OUTPUT:
                    int outIntensity = (int)((Math.max(-1, Math.min(1, ng.lastActivation)) + 1) / 2.0 * 255);
                    nodeColor = new Color(outIntensity, 220, outIntensity);
                    break;
                case BIAS:
                    nodeColor = new Color(255, 220, 80);
                    break;
                default:
                    int hIntensity = (int)((Math.max(-1, Math.min(1, ng.lastActivation)) + 1) / 2.0 * 255);
                    nodeColor = new Color(hIntensity, hIntensity, hIntensity);
                    break;
            }

            g2.setColor(nodeColor);
            g2.fillOval(pos[0] - nodeR, pos[1] - nodeR, nodeR * 2, nodeR * 2);
            g2.setColor(Color.BLACK);
            g2.drawOval(pos[0] - nodeR, pos[1] - nodeR, nodeR * 2, nodeR * 2);

            // Labels for inputs
            if (ng.type == NEATGenome.NodeType.INPUT) {
                int idx = inputs.indexOf(ng);
                if (idx >= 0 && idx < inputLabels.length) {
                    g2.setColor(Color.DARK_GRAY);
                    g2.setFont(new Font("SansSerif", Font.BOLD, 9));
                    FontMetrics fm = g2.getFontMetrics();
                    String lbl = inputLabels[idx];
                    g2.drawString(lbl, pos[0] - nodeR - fm.stringWidth(lbl) - 3, pos[1] + 4);
                }
            }
            // Label for output
            if (ng.type == NEATGenome.NodeType.OUTPUT) {
                g2.setColor(Color.DARK_GRAY);
                g2.setFont(new Font("SansSerif", Font.BOLD, 9));
                g2.drawString("Force", pos[0] + nodeR + 3, pos[1] + 4);
            }
            // Label for bias
            if (ng.type == NEATGenome.NodeType.BIAS) {
                g2.setColor(Color.DARK_GRAY);
                g2.setFont(new Font("SansSerif", Font.BOLD, 9));
                g2.drawString("Bias", pos[0] + nodeR + 3, pos[1] + 4);
            }
        }
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if (simState == SimulationState.WAITING_FOR_REPLAY) {
            repaint();
            return; // Pause physics until the replay button is clicked
        }
        
        // Speed up simulation by running multiple physics ticks per visual frame
        int ticksPerFrame = simSpeedSlider.getValue(); 
        for (int i = 0; i < ticksPerFrame; i++) {
            if (simState == SimulationState.TRAINING) {
                updatePhysics();
            } else if (simState == SimulationState.REPLAY) {
                updateReplay();
            } else {
                break;
            }
        }
        repaint();
    }

    private void drawFitnessHistogram(Graphics2D g2, int startX, int startY, int width, int height) {
        double[] data = previousGenerationFitness;
        boolean isLive = false;
        
        if (data == null) {
            if (population == null || population.isEmpty()) return;
            data = new double[population.size()];
            for (int i = 0; i < population.size(); i++) {
                data[i] = population.get(i).fitness;
            }
            isLive = true;
        }
        
        if (data.length == 0) return;

        double minFit = Double.MAX_VALUE;
        double maxFit = -Double.MAX_VALUE;
        for (double fit : data) {
            if (fit < minFit) minFit = fit;
            if (fit > maxFit) maxFit = fit;
        }

        if (minFit == maxFit) maxFit = minFit + 1; // Avoid division by zero

        int numBins = 20;
        int[] bins = new int[numBins];
        int maxFreq = 0;

        for (double fit : data) {
            int binIndex = (int) (((fit - minFit) / (maxFit - minFit)) * numBins);
            if (binIndex >= numBins) binIndex = numBins - 1;
            if (binIndex < 0) binIndex = 0;
            bins[binIndex]++;
            if (bins[binIndex] > maxFreq) maxFreq = bins[binIndex];
        }

        // Draw background
        g2.setColor(new Color(0, 0, 0, 180));
        g2.fillRect(startX, startY, width, height);

        // Draw title
        g2.setColor(Color.WHITE);
        g2.setFont(new Font("SansSerif", Font.BOLD, 12));
        g2.drawString(isLive ? "Fitness (Live Gen 1)" : "Fitness (Prev Gen)", startX + 10, startY + 20);

        // Draw bars
        int padX = 10;
        int padY = 30;
        int chartWidth = width - 2 * padX;
        int chartHeight = height - padY - 15;
        double binWidth = (double) chartWidth / numBins;

        for (int i = 0; i < numBins; i++) {
            int barHeight = maxFreq == 0 ? 0 : (int) (((double) bins[i] / maxFreq) * chartHeight);
            int barX = startX + padX + (int) (i * binWidth);
            int barY = startY + height - 15 - barHeight;

            g2.setColor(new Color(46, 204, 113, 200)); 
            g2.fillRect(barX, barY, (int) Math.max(1, binWidth - 1), barHeight);
        }

        // Draw labels
        g2.setColor(Color.LIGHT_GRAY);
        g2.setFont(new Font("SansSerif", Font.PLAIN, 10));
        g2.drawString(String.format("%.1f", minFit), startX + padX, startY + height - 3);
        
        String maxStr = String.format("%.1f", maxFit);
        FontMetrics fm = g2.getFontMetrics();
        g2.drawString(maxStr, startX + width - padX - fm.stringWidth(maxStr), startY + height - 3);
    }

    @Override
    public void keyPressed(KeyEvent e) {
        if (e.getKeyCode() == KeyEvent.VK_H) {
            showHistogram = !showHistogram;
        } else if (e.getKeyCode() == KeyEvent.VK_V) {
            showNN = !showNN;
            thicknessSlider.setVisible(showNN);
            thicknessLabel.setVisible(showNN);
        }
    }

    @Override public void keyReleased(KeyEvent e) {}
    @Override public void keyTyped(KeyEvent e) {}

    public static void main(String[] args) {
        JFrame frame = new JFrame("Neural Network Cartpole AI");
        Cartpole simulation = new Cartpole();
        
        frame.add(simulation);
        frame.setSize(800, 600);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    // SimulationContext implementation
    @Override public int pendulumMode()        { return pendulumMode; }
    @Override public boolean useAlternativeFitness() { return useAlternativeFitness; }
    @Override public boolean useCenterReward() { return useCenterReward; }
    @Override public double getCurrentGenAngleDeg() { return currentGenAngleDeg; }
}