def PROJECT_BASE_DIR = "/Volumes/Siyuan/Graham/NKI_512"

import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())


// Define output resolution
double requestedPixelSize = 0.25

// Convert to downsample
double downsample = requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()

annotations = getAnnotationObjects()
print annotations
removeObjects(annotations, true)

annotations.each{
    if(it.getPathClass() == null){return}
    addObject(it)

    className = it.getPathClass().toString()
    pathOutput = buildFilePath(PROJECT_BASE_DIR, name, className)
    mkdirs(pathOutput)
    new TileExporter(imageData)
        .downsample(downsample)     // Define export resolution
        .imageExtension('.png')     // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
        .tileSize(1024)              // Define size of each tile, in pixels
        //.labeledServer(labelServer) // Define the labeled image server to use (i.e. the one we just built)
        .annotatedCentroidTilesOnly(true)  // If true, only export tiles if there is a (labeled) annotation present
        //.includePartialTiles(false)
        .overlap(0)                // Define overlap, in pixel units at the export resolution
        .writeTiles(pathOutput)     // Write tiles to the specified directory
    removeObject(it, true)
    print 'Done!'
}

addObjects(annotations)
fireHierarchyUpdate()

print("All Done!")
