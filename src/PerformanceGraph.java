import javax.swing.*;
import java.awt.*;
import java.util.List;

/**
 * A panel that renders a line graph of fitness metrics over generations.
 */
public class PerformanceGraph extends JPanel {
    private List<GenerationData> history;
    private static final int PADDING = 50;
    private static final int LABEL_PADDING = 30;
    private int maxPoints = 100;

    public PerformanceGraph(List<GenerationData> history) {
        this.history = history;
        setBackground(new Color(250, 250, 250));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        synchronized(history) {
            if (history == null || history.isEmpty()) {
                drawEmptyMessage(g);
                return;
            }

            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            double maxFitness = getMaxFitness();
            int numGens = history.size();

            int width = getWidth();
            int height = getHeight();

            // Calculate graph bounds
            int graphWidth = width - 2 * PADDING - LABEL_PADDING;
            int graphHeight = height - 2 * PADDING - LABEL_PADDING;

            // Draw axes
            g2.setColor(Color.BLACK);
            g2.setStroke(new BasicStroke(2));
            // Y Axis
            g2.drawLine(PADDING + LABEL_PADDING, PADDING, PADDING + LABEL_PADDING, height - PADDING - LABEL_PADDING);
            // X Axis
            g2.drawLine(PADDING + LABEL_PADDING, height - PADDING - LABEL_PADDING, width - PADDING, height - PADDING - LABEL_PADDING);

            // Draw Labels
            g2.setFont(new Font("SansSerif", Font.PLAIN, 12));
            g2.drawString("Gen", width - PADDING, height - PADDING - LABEL_PADDING + 20);
            g2.drawString("Fitness", PADDING - 10, PADDING - 10);

            // Draw Y Axis Ticks and Labels
            int numTicks = 5;
            for (int i = 0; i <= numTicks; i++) {
                int y = height - PADDING - LABEL_PADDING - (i * graphHeight / numTicks);
                g2.drawLine(PADDING + LABEL_PADDING - 5, y, PADDING + LABEL_PADDING, y);
                String label = String.format("%.0f", i * maxFitness / numTicks);
                g2.drawString(label, PADDING, y + 5);
            }

            // Draw Shaded Distribution Area (Ribbon)
            if (numGens > 1) {
                drawDistributionArea(g2, history, graphWidth, graphHeight, maxFitness);
            }

            // Draw Data Lines
            if (numGens > 1) {
                drawMetricLine(g2, history, graphWidth, graphHeight, maxFitness, 0); // Best Fitness (Blue)
                drawMetricLine(g2, history, graphWidth, graphHeight, maxFitness, 1); // Median Fitness (Green)
                drawMetricLine(g2, history, graphWidth, graphHeight, maxFitness, 2); // Mean Fitness (Orange)
            }

            // Draw Legend
            drawLegend(g2, width - PADDING - 150, PADDING);
        }
    }

    private void drawDistributionArea(Graphics2D g2, List<GenerationData> history, int graphWidth, int graphHeight, double maxFitness) {
        int totalGens = history.size();
        int step = Math.max(1, totalGens / maxPoints);
        
        int drawCount = 0;
        for (int i = 0; i < totalGens; i += step) drawCount++;
        double xStep = (double) graphWidth / (drawCount - 1);

        Polygon ribbon = new Polygon();
        
        // Top boundary (Q3)
        int pointIndex = 0;
        for (int i = 0; i < totalGens; i += step) {
            int x = PADDING + LABEL_PADDING + (int) (pointIndex * xStep);
            int y = (int) (getHeight() - PADDING - LABEL_PADDING - (history.get(i).q3Fitness / maxFitness * graphHeight));
            ribbon.addPoint(x, y);
            pointIndex++;
        }
        
        // Bottom boundary (Q1) - loop backwards
        pointIndex = drawCount - 1;
        // Correcting loop for backwards iteration to match step points exactly
        int lastIndex = (drawCount - 1) * step;
        for (int i = lastIndex; i >= 0; i -= step) {
            int x = PADDING + LABEL_PADDING + (int) (pointIndex * xStep);
            int y = (int) (getHeight() - PADDING - LABEL_PADDING - (history.get(i).q1Fitness / maxFitness * graphHeight));
            ribbon.addPoint(x, y);
            pointIndex--;
        }

        g2.setColor(new Color(39, 174, 96, 40)); // Very transparent green
        g2.fill(ribbon);
    }

    private void drawMetricLine(Graphics2D g2, List<GenerationData> history, int graphWidth, int graphHeight, double maxFitness, int type) {
        int totalGens = history.size();
        
        // Downsampling: Use the user-defined maxPoints
        int step = Math.max(1, totalGens / maxPoints);
        
        // Effective number of points we will actually draw
        int drawCount = 0;
        for (int i = 0; i < totalGens; i += step) drawCount++;
        
        double xStep = (double) graphWidth / (drawCount - 1);

        Color color;
        if (type == 0) color = new Color(41, 128, 185);      // Best (Blue)
        else if (type == 1) color = new Color(39, 174, 96); // Median (Green)
        else color = new Color(230, 126, 34);               // Mean (Orange)

        g2.setColor(color);
        g2.setStroke(new BasicStroke(2.5f));

        int prevX = -1, prevY = -1;
        int pointIndex = 0;

        for (int i = 0; i < totalGens; i += step) {
            double val;
            if (type == 0) val = history.get(i).bestFitness;
            else if (type == 1) val = history.get(i).medianFitness;
            else val = history.get(i).meanFitness;

            int x = PADDING + LABEL_PADDING + (int) (pointIndex * xStep);
            int y = (int) (getHeight() - PADDING - LABEL_PADDING - (val / maxFitness * graphHeight));

            if (prevX != -1) {
                g2.drawLine(prevX, prevY, x, y);
            }
            
            // Draw node (only if we aren't too crowded)
            if (drawCount < 50) {
                g2.fillOval(x - 3, y - 3, 6, 6);
            } else if (drawCount < 150) {
                g2.fillOval(x - 1, y - 1, 2, 2);
            }

            prevX = x;
            prevY = y;
            pointIndex++;
        }
    }

    private void drawLegend(Graphics2D g2, int x, int y) {
        g2.setFont(new Font("SansSerif", Font.BOLD, 12));
        
        g2.setColor(new Color(41, 128, 185));
        g2.fillRect(x, y, 15, 15);
        g2.setColor(Color.BLACK);
        g2.drawString("Best Fitness", x + 20, y + 12);

        g2.setColor(new Color(39, 174, 96));
        g2.fillRect(x, y + 25, 15, 15);
        g2.setColor(Color.BLACK);
        g2.drawString("Median Fitness", x + 20, y + 37);

        g2.setColor(new Color(230, 126, 34));
        g2.fillRect(x, y + 50, 15, 15);
        g2.setColor(Color.BLACK);
        g2.drawString("Mean Fitness (Avg)", x + 20, y + 62);

        g2.setColor(new Color(39, 174, 96, 80));
        g2.fillRect(x, y + 75, 15, 15);
        g2.setColor(Color.BLACK);
        g2.drawString("Spread (Q1-Q3)", x + 20, y + 87);
    }

    private void drawEmptyMessage(Graphics g) {
        g.setFont(new Font("SansSerif", Font.BOLD, 18));
        g.setColor(Color.GRAY);
        FontMetrics fm = g.getFontMetrics();
        String msg = "Not enough data yet. Complete more generations!";
        g.drawString(msg, (getWidth() - fm.stringWidth(msg)) / 2, getHeight() / 2);
    }

    private double getMaxFitness() {
        synchronized(history) {
            double max = 1.0;
            for (GenerationData data : history) {
                if (data.bestFitness > max) max = data.bestFitness;
            }
            return max * 1.1; // 10% buffer
        }
    }

    public void setMaxPoints(int maxPoints) {
        this.maxPoints = maxPoints;
        repaint();
    }

    public static JFrame showGraph(List<GenerationData> history, double[] currentFitness) {
        JFrame frame = new JFrame("Performance Trend & Distribution");
        frame.setSize(1000, 600); 
        frame.setLayout(new BorderLayout());
        
        JSplitPane splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        splitPane.setDividerLocation(650);
        
        PerformanceGraph graph = new PerformanceGraph(history);
        FitnessHistogram histogram = new FitnessHistogram(currentFitness);
        
        splitPane.setLeftComponent(graph);
        splitPane.setRightComponent(histogram);
        
        frame.add(splitPane, BorderLayout.CENTER);
        
        // Store the histogram in the frame's client properties or return a wrapper
        // But since we return the frame, Cartpole can look up the components
        frame.getRootPane().putClientProperty("graphPanel", graph);
        frame.getRootPane().putClientProperty("histogramPanel", histogram);
        
        // Slider for controlling data points
        JPanel controls = new JPanel(new FlowLayout());
        JLabel label = new JLabel("History Data Points: 100");
        JSlider slider = new JSlider(10, 500, 100);
        slider.addChangeListener(e -> {
            graph.setMaxPoints(slider.getValue());
            label.setText("History Data Points: " + slider.getValue());
        });
        
        controls.add(new JLabel("Detail Level: "));
        controls.add(slider);
        controls.add(label);
        frame.add(controls, BorderLayout.SOUTH);
        
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        return frame;
    }
}
