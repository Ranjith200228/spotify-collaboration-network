# Best-effort fetch of SpotifyFeatures.csv (Kaggle "ultimate-spotify-tracks-db" schema).
# Tries several public mirrors. Falls back to a HuggingFace dataset on failure.
[CmdletBinding()]
param(
    [string]$OutDir = "data/raw"
)

$ErrorActionPreference = 'Continue'
$dest = Join-Path $OutDir "SpotifyFeatures.csv"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$mirrors = @(
    'https://raw.githubusercontent.com/HrushikeshChaudhari/Spotify-Features-Recommender-System/master/SpotifyFeatures.csv',
    'https://raw.githubusercontent.com/krishna321-stack/Spotify-Recommendation-System/main/SpotifyFeatures.csv',
    'https://media.githubusercontent.com/media/devarsh10/Spotify-music-recommendation-system/main/SpotifyFeatures.csv',
    'https://raw.githubusercontent.com/Spotify-tracks-analysis/data/main/SpotifyFeatures.csv',
    'https://raw.githubusercontent.com/sjapko/SpotifyAnalysis/master/SpotifyFeatures.csv',
    'https://raw.githubusercontent.com/yamaceay/spotify-genre-analysis/main/SpotifyFeatures.csv'
)

foreach ($u in $mirrors) {
    Write-Host "trying $u"
    try {
        Invoke-WebRequest -Uri $u -OutFile $dest -UseBasicParsing -TimeoutSec 60 -ErrorAction Stop
        $f = Get-Item $dest -ErrorAction SilentlyContinue
        if ($f -and $f.Length -gt 5000000) {
            Write-Host "  OK size=$($f.Length) from $u"
            exit 0
        } else {
            Write-Host "  size too small ($($f.Length)); removing"
            Remove-Item $dest -Force -ErrorAction SilentlyContinue
        }
    } catch {
        Write-Host "  fail: $_"
    }
}

Write-Host "All SpotifyFeatures.csv mirrors failed; falling back to HuggingFace alternative dataset."
$hf = 'https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv'
try {
    $altDest = Join-Path $OutDir "hf_spotify_tracks.csv"
    Invoke-WebRequest -Uri $hf -OutFile $altDest -UseBasicParsing -TimeoutSec 90 -ErrorAction Stop
    $f = Get-Item $altDest -ErrorAction SilentlyContinue
    if ($f -and $f.Length -gt 1000000) {
        Write-Host "  HF OK size=$($f.Length) at $altDest"
        exit 2
    }
} catch {
    Write-Host "  HF fail: $_"
}

exit 1
