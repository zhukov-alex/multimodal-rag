{{ if .System }}
System:
{{ .System }}
{{ end }}

{{ if .Prompt }}
User Input:
{{ .Prompt }}
{{ end }}

{{ if .Images }}
{{ range $index, $img := .Images }}
[Visual context {{ $index }} inserted above]
{{ end }}
{{ end }}

Assistant:
