package main

import (
	"strings"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"
)

func main() {
	app := app.New()
	win := app.NewWindow("Remove Newlines")

	textInput := widget.NewEntry()
	textInput.MultiLine = true
	textInput.SetPlaceHolder("Input text here")

	textOutput := widget.NewEntry()
	textOutput.MultiLine = true
	textOutput.SetPlaceHolder("Output text here")
	textOutput.Disable()

	removeNewlines := func() {
		inputText := textInput.Text
		outputText := strings.ReplaceAll(inputText, "\n", " ")
		textOutput.SetText(outputText)
	}

	clearInput := func() {
		textInput.SetText("")
	}

	removeButton := widget.NewButton("Remove Newlines", removeNewlines)
	clearButton := widget.NewButton("Clear Input", clearInput)

	buttons := container.NewHBox(removeButton, clearButton)

	content := container.NewVBox(
		textInput,
		buttons,
		textOutput,
		)

	win.SetContent(content)
	win.Resize(fyne.NewSize(500, 400))
	win.ShowAndRun()
}
