import java.awt.Dimension
import java.awt.Toolkit
import java.awt.datatransfer.Clipboard
import java.awt.datatransfer.StringSelection
import javax.swing.*

fun main() {
    SwingUtilities.invokeLater {
        val frame = JFrame("Remove Newlines")
        frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE

        val inputTextArea = JTextArea()
        val outputTextArea = JTextArea()

        fun removeNewlines() {
            val inputText = inputTextArea.text
            val outputText = inputText.replace("\n", " ")
            outputTextArea.text = outputText
        }

        fun clearInput() {
            inputTextArea.text = ""
        }

        fun copyOutputToClipboard() {
            val outputText = outputTextArea.text
            val stringSelection = StringSelection(outputText)
            val clipboard: Clipboard = Toolkit.getDefaultToolkit().systemClipboard
            clipboard.setContents(stringSelection, null)
        }

        val removeButton = JButton("Remove Newlines")
        removeButton.addActionListener { removeNewlines() }

        val clearButton = JButton("Clear Input")
        clearButton.addActionListener { clearInput() }

        val copyButton = JButton("Copy Output")
        copyButton.addActionListener { copyOutputToClipboard() }

        val buttonPanel = JPanel()
        buttonPanel.add(removeButton)
        buttonPanel.add(clearButton)
        buttonPanel.add(copyButton)

        val contentPane = frame.contentPane
        contentPane.layout = BoxLayout(contentPane, BoxLayout.Y_AXIS)
        contentPane.add(JScrollPane(inputTextArea))
        contentPane.add(buttonPanel)
        contentPane.add(JScrollPane(outputTextArea))

        frame.pack()
        frame.size = Dimension(500, 400)
        frame.setLocationRelativeTo(null)
        frame.isVisible = true
    }
}
