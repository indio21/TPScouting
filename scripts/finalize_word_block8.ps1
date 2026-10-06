param(
    [string]$Source = "C:\Tesis\TPScouting\docs\revision_octubre_2026\word\TRABAJO_FINAL_TPScouting_CORREGIDO_BLOQUE7_2026-10-06.docx",
    [string]$Output = "C:\Tesis\TPScouting\docs\revision_octubre_2026\word\TRABAJO_FINAL_TPScouting_CANDIDATO_FINAL_BLOQUE8_2026-10-06.docx",
    [string]$Pdf = "C:\Tesis\TPScouting\docs\revision_octubre_2026\pdf\TRABAJO_FINAL_TPScouting_CANDIDATO_FINAL_BLOQUE8_2026-10-06.pdf",
    [string]$ProofingReport = "C:\Tesis\TPScouting\docs\revision_octubre_2026\revision_gramatical_word_bloque8.txt",
    [switch]$SkipIndexUpdate
)

$ErrorActionPreference = "Stop"
$word = $null
$document = $null

function Release-ComObject([object]$Object) {
    if ($null -ne $Object) {
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($Object)
    }
}

try {
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "No existe el documento fuente: $Source"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Pdf) | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Output -Force

    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($Output, $false, $false)
    Write-Output "STEP=open"

    if (-not $SkipIndexUpdate) {
        foreach ($toc in $document.TablesOfContents) { [void]$toc.Update() }
        Write-Output "STEP=toc"
        foreach ($tof in $document.TablesOfFigures) { [void]$tof.Update() }
        Write-Output "STEP=tof"
    }

    $document.Repaginate()
    Write-Output "STEP=repaginate"
    $pages = $document.ComputeStatistics(2)

    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("Revision local de Word - Bloque 8")
    $lines.Add("Documento: $Output")
    $lines.Add("Paginas despues de actualizar campos: $pages")
    $lines.Add("La enumeracion automatizada de errores de Word se omitio: la API COM no termino en 120 segundos.")
    $lines.Add("La revision linguistica reproducible se registra por separado.")
    [IO.File]::WriteAllLines($ProofingReport, $lines, [Text.UTF8Encoding]::new($false))

    $document.Save()
    Write-Output "STEP=save"
    $document.ExportAsFixedFormat($Pdf, 17)
    Write-Output "STEP=pdf"
    $document.Close($false)
    $document = $null
    $word.Quit()
    $word = $null

    $docHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Output).Hash
    $pdfHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Pdf).Hash
    Write-Output "PAGES=$pages"
    Write-Output "DOCX_SHA256=$docHash"
    Write-Output "PDF_SHA256=$pdfHash"
    Write-Output "DOCX=$Output"
    Write-Output "PDF=$Pdf"
    Write-Output "PROOFING=$ProofingReport"
}
finally {
    if ($null -ne $document) {
        try { $document.Close($false) } catch {}
        Release-ComObject $document
    }
    if ($null -ne $word) {
        try { $word.Quit() } catch {}
        Release-ComObject $word
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
