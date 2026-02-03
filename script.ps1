$text = "Hello World"
$colors = @('Red', 'Yellow', 'Green', 'Cyan', 'Blue', 'Magenta')

Clear-Host
Write-Host "`n"

for ($i = 0; $i -lt 20; $i++) {
    $colorIndex = $i % $colors.Count
    Write-Host "`r  $text" -ForegroundColor $colors[$colorIndex] -NoNewline
    Start-Sleep -Milliseconds 200
}

Write-Host "`n`n"
