import javax.swing.*;
import java.awt.*;
import java.util.Arrays;

/**
 * A panel that renders a histogram of fitness values for the current population.
 */
public class FitnessHistogram extends JPanel {
    private double[] fitnessData;
    private static final int PADDING = 40;

    public FitnessHistogram(double[] fitnessData) {
        this.fitnessData = fitnessData;
        setBackground(new Color(250, 250, 250));
    }

    public void updateData(double[] newData) {
        this.fitnessData = newData;
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (fitnessData == null || fitnessData.length == 0) {
            drawEmptyMessage(g);
            return;
        }

        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        double minFit = Double.MAX_VALUE;
        double maxFit = -Double.MAX_VALUE;
        for (double fit : fitnessData) {
            if (fit < minFit) minFit = fit;
            if (fit > maxFit) maxFit = fit;
        }

        if (minFit == maxFit) maxFit = minFit + 1;

        int numBins = 20;
        int[] bins = new int[numBins];
        int maxFreq = 0;

        for (double fit : fitnessData) {
            int binIndex = (int) (((fit - minFit) / (maxFit - minFit)) * numBins);
            if (binIndex >= numBins) binIndex = numBins - 1;
            if (binIndex < 0) binIndex = 0;
            bins[binIndex]++;
            if (bins[binIndex] > maxFreq) maxFreq = bins[binIndex];
        }

        int width = getWidth() - 2 * PADDING;
        int height = getHeight() - 2 * PADDING;
        int binWidth = width / numBins;

        // Draw Bars
        for (int i = 0; i < numBins; i++) {
            int binHeight = (int) (((double) bins[i] / maxFreq) * height);
            int x = PADDING + i * binWidth;
            int y = getHeight() - PADDING - binHeight;

            // Color gradient based on bin index (worse to better)
            float hue = (float) i / numBins * 0.3f + 0.3f; // Greenish range
            g2.setColor(Color.getHSBColor(hue, 0.7f, 0.8f));
            g2.fillRect(x, y, binWidth - 2, binHeight);
            
            g2.setColor(new Color(0, 0, 0, 50));
            g2.drawRect(x, y, binWidth - 2, binHeight);
        }

        // Draw Axes
        g2.setColor(Color.BLACK);
        g2.drawLine(PADDING, getHeight() - PADDING, getWidth() - PADDING, getHeight() - PADDING); // X
        g2.drawLine(PADDING, PADDING, PADDING, getHeight() - PADDING); // Y

        // Draw Labels
        g2.setFont(new Font("SansSerif", Font.PLAIN, 10));
        g2.drawString(String.format("%.1f", minFit), PADDING, getHeight() - PADDING + 15);
        g2.drawString(String.format("%.1f", maxFit), getWidth() - PADDING - 30, getHeight() - PADDING + 15);
        g2.drawString("Fitness Distribution", (getWidth() - 100) / 2, PADDING / 2);
    }

    private void drawEmptyMessage(Graphics g) {
        g.setFont(new Font("SansSerif", Font.BOLD, 14));
        g.setColor(Color.GRAY);
        g.drawString("Waiting for generation data...", 20, getHeight() / 2);
    }
}
