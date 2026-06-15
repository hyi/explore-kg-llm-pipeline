{{- define "explorer.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "explorer.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "explorer.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" -}}
{{- end -}}

{{- define "explorer.labels" -}}
helm.sh/chart: {{ include "explorer.chart" . }}
{{ include "explorer.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "explorer.selectorLabels" -}}
app.kubernetes.io/name: {{ include "explorer.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "explorer.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "explorer.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "explorer.secretName" -}}
{{- if .Values.env.existingSecret -}}
{{- .Values.env.existingSecret -}}
{{- else -}}
{{- printf "%s-env" (include "explorer.fullname" .) -}}
{{- end -}}
{{- end -}}
