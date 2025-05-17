import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { $el } from "/scripts/ui.js";

function imageDataToUrl(data) {
    return api.apiURL(`/view?filename=${encodeURIComponent(data.filename)}&type=${data.type}&subfolder=${data.subfolder}${app.getPreviewFormatParam()}${app.getRandParam()}`);
}

function measureText(ctx, str) {
    return ctx.measureText(str).width;
}

app.registerExtension({
    name: "ComfyUI-ImageCompare",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Image Compare (🐺)") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                console.log("[ComfyUI-ImageCompare] onNodeCreated - Initializing node properties and state."); // DEBUG

                if (!this.properties || !("comparer_mode" in this.properties)) {
                    this.properties = { ...this.properties, "comparer_mode": "Slide" };
                }

                this.imageIndex = 0;
                this.isPointerDown = false;
                this.isPointerOver = false;
                this.pointerOverPos = [0, 0];

                this.compare_value = { images: [] };
                this.compare_selected = [];
                this.compare_hitAreas = {};
                console.log("[ComfyUI-ImageCompare] onNodeCreated - State initialized:", this.compare_value); // DEBUG

                const onExecuted = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (output) {
                    console.log("[ComfyUI-ImageCompare] onExecuted - Function Entered."); // DEBUG - Moved to top
                    console.log("[ComfyUI-ImageCompare] onExecuted - Received output from Python:", output); // DEBUG - Log raw output first
                    try {
                        onExecuted?.apply(this, arguments); // Call original
                        // Pass the raw output 'ui' data (containing a_images/b_images) to the node's setValue method
                        // Check if the received output itself has the expected keys
                        if (output && (output.a_images || output.b_images)) {
                            console.log("[ComfyUI-ImageCompare] onExecuted - Calling setValue with received output:", output); // DEBUG
                            this.setValue(output); // Pass the output directly
                        } else {
                            console.warn("[ComfyUI-ImageCompare] onExecuted - Received data does not contain a_images or b_images.", output); // DEBUG
                        }
                        // Ensure redraw after execution and potential image loading
                        this.setDirtyCanvas(true, true);
                    } catch (error) {
                        console.error("[ComfyUI-ImageCompare] onExecuted - Error during execution:", error); // DEBUG
                    }
                };

                const computeSize = nodeType.prototype.computeSize;
                nodeType.prototype.computeSize = function(out) {
                    let size = computeSize?.apply(this, arguments);
                    if (!size) size = [200, 80]; // Default size if base computeSize fails
                    // console.log(`[ComfyUI-ImageCompare] computeSize - Base size: ${size[0]}x${size[1]}`); // DEBUG SIZE

                    const margin = 10;
                    let requiredHeight = LiteGraph.NODE_TITLE_HEIGHT;

                     // Add height for selectors if needed (Check if compare_value exists)
                    if (this.compare_value && this.compare_value.images?.length > 2) {
                        requiredHeight += 20 + margin; // Selector height + margin
                    } else {
                        requiredHeight += margin / 2;
                    }

                    // Estimate image height (Check if compare_selected exists)
                    let imageHeight = 200; // Default/minimum image height
                    if (this.compare_selected && this.compare_selected.length > 0 && this.compare_selected[0].img?.naturalWidth) {
                        const img = this.compare_selected[0].img;
                        const imageAspectRatio = img.naturalWidth / img.naturalHeight;
                         // Calculate based on current node width (size[0])
                        const availableWidth = size[0] - margin * 2;
                        imageHeight = availableWidth / imageAspectRatio;
                        imageHeight = Math.max(20, imageHeight); // Min height
                     } else if (this.properties?.height) { // This might be unreliable
                         imageHeight = Math.max(20, this.properties.height - requiredHeight - margin);
                         // console.log(`[ComfyUI-ImageCompare] computeSize - Using properties height fallback: ${imageHeight}`); // DEBUG
                     } else {
                         // Fallback if no image loaded yet - Use remaining size or a default
                         imageHeight = Math.max(20, (size[1] || 80) - requiredHeight - margin);
                         // console.log(`[ComfyUI-ImageCompare] computeSize - Using fallback height: ${imageHeight}`); // DEBUG
                     }

                    requiredHeight += imageHeight + margin; // Add image height and bottom margin

                    // Ensure minimum height is met
                    size[1] = Math.max(size[1] || 80, requiredHeight);

                    // console.log(`[ComfyUI-ImageCompare] computeSize - Calculated size: ${size[0]}x${size[1]}`); // DEBUG SIZE
                    // console.log(`[ComfyUI-ImageCompare] computeSize - Final requiredHeight: ${requiredHeight}`); // DEBUG
                    return size;
                }

                const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
                nodeType.prototype.getExtraMenuOptions = function(_, options) {
                    getExtraMenuOptions?.apply(this, arguments);
                    const imageIndex = (this.pointerOverPos && this.pointerOverPos[0] > this.size[0] / 2) ? 1 : 0;
                    const selectedImage = this.compare_selected[imageIndex];

                    if (selectedImage?.img) {
                        options.unshift(
                            {
                                content: "Open Image",
                                 callback: () => { window.open(selectedImage.url, "_blank"); },
                            },
                            { content: "Save Image", callback: () => { } },
                            { content: "Copy Image", callback: () => { } }
                        );
                    }
                };

                const onMouseDown = nodeType.prototype.onMouseDown;
                nodeType.prototype.onMouseDown = function(event, pos, canvas) {
                     onMouseDown?.apply(this, arguments);
                     if (this.compare_hitAreas?.image &&
                         pos[0] >= this.compare_hitAreas.image[0] && pos[0] <= this.compare_hitAreas.image[0] + this.compare_hitAreas.image[2] &&
                         pos[1] >= this.compare_hitAreas.image[1] && pos[1] <= this.compare_hitAreas.image[1] + this.compare_hitAreas.image[3])
                     {
                        this.isPointerDown = true;
                        if (this.properties["comparer_mode"] === "Click") {
                            this.imageIndex = 1 - this.imageIndex;
                            this.setDirtyCanvas(true, false);
                        }
                     }
                };

                const onMouseMove = nodeType.prototype.onMouseMove;
                nodeType.prototype.onMouseMove = function(event, pos, canvas) {
                    onMouseMove?.apply(this, arguments);
                    // console.log(`[ComfyUI-ImageCompare] onMouseMove - Pos: ${pos[0]}, ${pos[1]}`); // DEBUG
                    this.pointerOverPos = [...pos];
                    if (this.isPointerDown && this.properties["comparer_mode"] === "Slide") {
                        this.setDirtyCanvas(true, false);
                    } else if (this.isPointerOver && this.properties["comparer_mode"] === "Click") {
                    }
                };

                const onMouseEnter = nodeType.prototype.onMouseEnter;
                nodeType.prototype.onMouseEnter = function(event) {
                    onMouseEnter?.apply(this, arguments);
                    this.isPointerOver = true;
                    this.isPointerDown = !!app.canvas.pointer_is_down;
                     this.setDirtyCanvas(true, false);
                };

                 const onMouseLeave = nodeType.prototype.onMouseLeave;
                 nodeType.prototype.onMouseLeave = function(event) {
                     onMouseLeave?.apply(this, arguments);
                     this.isPointerOver = false;
                     this.isPointerDown = false;
                     this.setDirtyCanvas(true, false);
                };

                const onMouseUp = nodeType.prototype.onMouseUp;
                nodeType.prototype.onMouseUp = function(event, pos, canvas) {
                     if (this.isPointerDown) {
                         this.isPointerDown = false;
                         this.setDirtyCanvas(true, false);
                     }
                     onMouseUp?.apply(this, arguments);
                };

                nodeData.properties = nodeData.properties || {};
                nodeData.properties["comparer_mode"] = ["Slide", "Click"];

                nodeType.prototype.onDrawForeground = function(ctx) {
                    if (this.flags.collapsed) return;
                    console.log(`[ComfyUI-ImageCompare] onDrawForeground - Node size: ${this.size[0]}x${this.size[1]}`); // DEBUG SIZE

                    this.compare_hitAreas = {};
                    const margin = 10;
                    const titleHeight = LiteGraph.NODE_TITLE_HEIGHT;
                    let y = titleHeight;
                    const widgetWidth = this.size[0];

                    if (this.compare_value.images.length > 2) {
                        ctx.save();
                        ctx.textAlign = "left";
                        ctx.textBaseline = "top";
                        ctx.font = `14px Arial`;
                        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
                        const textY = y + margin / 2;
                        let currentX = margin;
                        const selectorHeight = 20;

                        this.compare_value.images.forEach((imgData, index) => {
                            const text = imgData.name || `Image ${index + 1}`;
                            const textWidth = measureText(ctx, text);
                            const boxWidth = textWidth + 20;
                            const boxRect = [currentX, textY, boxWidth, selectorHeight];

                            ctx.fillStyle = imgData.selected ? LiteGraph.WIDGET_BGCOLOR_TRUE : LiteGraph.WIDGET_BGCOLOR;
                            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
                            ctx.strokeRect(...boxRect);
                            ctx.fillRect(...boxRect);

                            ctx.fillStyle = imgData.selected ? LiteGraph.WIDGET_TEXT_COLOR_TRUE : LiteGraph.WIDGET_TEXT_COLOR;
                            ctx.fillText(text, currentX + 10, textY + selectorHeight / 2 - 7);

                            this.compare_hitAreas[`select_${index}`] = boxRect;
                            currentX += boxWidth + 5;
                        });
                        ctx.restore();
                        y += selectorHeight + margin;
                    } else {
                        y += margin / 2;
                    }

                    const availableWidth = widgetWidth - margin * 2;
                    const availableHeight = this.size[1] - y - margin;
                    let drawWidth = availableWidth;
                    let drawHeight = 20;

                    if (this.compare_selected.length > 0 && this.compare_selected[0].img?.naturalWidth) {
                        const imgA = this.compare_selected[0].img;
                        const imgB = this.compare_selected.length > 1 ? this.compare_selected[1]?.img : null;
                        const mode = this.properties["comparer_mode"] || "Slide";

                        const imageAspectRatio = imgA.naturalWidth / imgA.naturalHeight;
                        drawWidth = availableWidth;
                        drawHeight = drawWidth / imageAspectRatio;

                        if (drawHeight > availableHeight) {
                            drawHeight = availableHeight;
                            drawWidth = drawHeight * imageAspectRatio;
                        }
                        if (drawHeight <= 0) drawHeight = 20;

                        const drawX = (widgetWidth - drawWidth) / 2;
                        const drawY = y;

                        let clipWidth = drawWidth;
                        if (mode === "Slide" && this.isPointerOver && imgB) {
                            const relativeX = this.pointerOverPos[0] - drawX;
                            clipWidth = Math.max(0, Math.min(drawWidth, relativeX));
                        } else if (mode === "Click" && this.imageIndex === 1 && imgB) {
                            ctx.drawImage(imgB, drawX, drawY, drawWidth, drawHeight);
                        } else {
                            ctx.drawImage(imgA, drawX, drawY, drawWidth, drawHeight);
                        }

                        if (mode === "Slide" && imgB) {
                            // Draw imgB first, then imgA clipped on top
                            ctx.drawImage(imgB, drawX, drawY, drawWidth, drawHeight);
                            
                            ctx.save();
                            ctx.beginPath();
                            ctx.rect(drawX, drawY, clipWidth, drawHeight); // Clip rect starts at drawX
                            ctx.clip();
                            ctx.drawImage(imgA, drawX, drawY, drawWidth, drawHeight); 
                            ctx.restore();

                            ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(drawX + clipWidth, drawY);
                            ctx.lineTo(drawX + clipWidth, drawY + drawHeight);
                            ctx.stroke();
                        } else if (mode !== "Click" || this.imageIndex === 0) {
                            ctx.drawImage(imgA, drawX, drawY, drawWidth, drawHeight);
                        }

                        this.compare_hitAreas["image"] = [drawX, drawY, drawWidth, drawHeight];
                    } else {
                        const drawX = margin;
                        const drawY = y;
                        drawWidth = availableWidth;
                        drawHeight = availableHeight > 20 ? availableHeight : 20;
                        ctx.fillStyle = LiteGraph.NODE_DEFAULT_BGCOLOR;
                        ctx.fillRect(drawX, drawY, drawWidth, drawHeight);
                        this.compare_hitAreas["image"] = [drawX, drawY, drawWidth, drawHeight];
                    }

                    if (this.isPointerOver && this.properties["comparer_mode"] === "Slide") {
                       this.setDirtyCanvas(true, false);
                    }
                };

                nodeType.prototype.setValue = function (v) {
                    console.log("[ComfyUI-ImageCompare] setValue - Function Entered."); // DEBUG
                    let cleanedVal;
                    console.log("[ComfyUI-ImageCompare] setValue - Raw input 'v':", v); // DEBUG - Simpler log
                    if (Array.isArray(v?.images)) { // Check if v has images array
                        console.log("[ComfyUI-ImageCompare] setValue - Processing input as v.images array."); // DEBUG
                        cleanedVal = v.images;
                    } else if (Array.isArray(v?.a_images) || Array.isArray(v?.b_images)) { // Handle old format
                        console.log("[ComfyUI-ImageCompare] setValue - Processing input as v.a_images/b_images."); // DEBUG
                        const a_images = v.a_images || [];
                        const b_images = v.b_images || [];
                        cleanedVal = [];
                        const multiple = a_images.length + b_images.length > 2;
                        console.log(`[ComfyUI-ImageCompare] setValue - Found ${a_images.length} A images, ${b_images.length} B images.`); // DEBUG
                        a_images.forEach((d, i) => cleanedVal.push({
                            url: imageDataToUrl(d),
                            name: a_images.length > 1 || multiple ? `A${i+1}` : "A",
                            selected: false, // Default to false, select below
                        }));
                        b_images.forEach((d, i) => cleanedVal.push({
                            url: imageDataToUrl(d),
                            name: b_images.length > 1 || multiple ? `B${i+1}` : "B",
                            selected: false, // Default to false, select below
                        }));
                        console.log("[ComfyUI-ImageCompare] setValue - Built cleanedVal:", cleanedVal); // DEBUG
                    } else {
                        console.log("[ComfyUI-ImageCompare] setValue - Input format not recognized."); // DEBUG
                        cleanedVal = []; // No valid image data
                    }

                    console.log("[ComfyUI-ImageCompare] setValue - Entering selection logic."); // DEBUG
                    // Ensure we always have exactly two selected images if possible
                    let currentSelected = cleanedVal.filter((d) => d.selected);

                    // If nothing selected, select first two (or one if only one exists)
                    if (currentSelected.length === 0 && cleanedVal.length > 0) {
                        console.log("[ComfyUI-ImageCompare] setValue - Selecting first 1 or 2 images."); // DEBUG
                        cleanedVal[0].selected = true;
                        if (cleanedVal.length > 1) {
                             cleanedVal[1].selected = true;
                        }
                    }
                    // If only one selected, select the next available one
                    else if (currentSelected.length === 1 && cleanedVal.length > 1) {
                        console.log("[ComfyUI-ImageCompare] setValue - Only 1 selected, selecting next available."); // DEBUG
                         const firstUnselected = cleanedVal.find(d => !d.selected);
                         if (firstUnselected) {
                             firstUnselected.selected = true;
                         }
                    }
                     // If more than two selected, deselect extras (keep first two found)
                    else if (currentSelected.length > 2) {
                        console.log("[ComfyUI-ImageCompare] setValue - More than 2 selected, deselecting extras."); // DEBUG
                        let count = 0;
                        cleanedVal.forEach(d => {
                            if (d.selected) {
                                count++;
                                if (count > 2) {
                                    d.selected = false;
                                }
                            }
                        });
                    }

                    console.log("[ComfyUI-ImageCompare] setValue - Exiting selection logic."); // DEBUG
                    this.compare_value.images = cleanedVal;
                    this.compare_selected = this.compare_value.images.filter((d) => d.selected);

                    console.log("[ComfyUI-ImageCompare] setValue - Final this.compare_value.images:", this.compare_value.images); // DEBUG
                    console.log("[ComfyUI-ImageCompare] setValue - Final this.compare_selected:", this.compare_selected); // DEBUG
                    // Load image objects for selected items
                    this.loadSelectedImages();
                };

                nodeType.prototype.loadSelectedImages = function() {
                    console.log(`[ComfyUI-ImageCompare] loadSelectedImages - Attempting to load ${this.compare_selected.length} images.`); // DEBUG
                    this.compare_selected.forEach(sel => {
                        if (!sel.img && sel.url) {
                            console.log(`[ComfyUI-ImageCompare] loadSelectedImages - Creating Image() for URL: ${sel.url}`); // DEBUG
                            sel.img = new Image();
                            sel.img.src = sel.url;
                            sel.img.onload = () => { 
                                console.log(`[ComfyUI-ImageCompare] loadSelectedImages - Image loaded: ${sel.url}`); // DEBUG
                                this.setDirtyCanvas(true, false); 
                            }; // Redraw when loaded
                            sel.img.onerror = (err) => { 
                                console.error(`[ComfyUI-ImageCompare] loadSelectedImages - Error loading image: ${sel.url}`, err); // DEBUG
                            }; 
                        }
                    });
                };

                nodeType.prototype.handleSelectionClick = function(index) {
                    console.log(`[ComfyUI-ImageCompare] handleSelectionClick - Clicked index: ${index}`); // DEBUG
                    const clickedImageData = this.compare_value.images[index];
                    if (!clickedImageData) return;

                    let selectedCount = this.compare_value.images.filter(d => d.selected).length;

                    if (clickedImageData.selected) {
                        if (selectedCount > 1) {
                            clickedImageData.selected = false;
                            selectedCount--;
                        }
                    } else {
                        clickedImageData.selected = true;
                        selectedCount++;
                        if (selectedCount > 2) {
                            const firstSelected = this.compare_value.images.find(d => d.selected && d !== clickedImageData);
                            if (firstSelected) {
                                firstSelected.selected = false;
                                selectedCount--;
                            }
                        }
                    }

                    if (selectedCount === 0 && this.compare_value.images.length > 0) {
                         this.compare_value.images[0].selected = true;
                    }

                    this.compare_selected = this.compare_value.images.filter((d) => d.selected);
                    this.loadSelectedImages();
                    this.setDirtyCanvas(true, false);
                    console.log(`[ComfyUI-ImageCompare] handleSelectionClick - New selected:`, JSON.parse(JSON.stringify(this.compare_selected))); // DEBUG
                };

                nodeType.prototype.getHelp = function() {
                    return $el("div", [
                        $el("p", [`The ${this.title} node compares two images (A and B) side-by-side.`]),
                        $el("ul", [
                            $el("li", [
                                $el("strong", ["Mode:"]),
                                $el("ul", [
                                    $el("li", [$el("code", ["Slide:"]), " Drag the mouse left/right over the image to reveal image B under image A."]),
                                    $el("li", [$el("code", ["Click:"]), " Click on the image to toggle between showing image A and image B."])
                                ])
                            ]),
                            $el("li", [
                                $el("strong", ["Inputs:"]),
                                $el("ul", [
                                    $el("li", [$el("code", ["image_a"]), " (Optional): The first image (or batch)."]),
                                    $el("li", [$el("code", ["image_b"]), " (Optional): The second image (or batch)."])
                                ])
                            ]),
                            $el("li", [
                                $el("strong", ["Selection:"]),
                                $el("p", ["If inputs are batches, selection boxes appear above the image. Click to choose which two images from the combined batches to compare."])
                            ]),
                             $el("li", [
                                $el("strong", ["Right-Click Menu:"]),
                                $el("p", ["Right-clicking on the image area provides options (Open, Save, Copy) for the currently relevant image (A or B depending on mode/interaction)."])
                            ]),
                        ])
                    ]).outerHTML;
                };
            }
        }
    },

    nodeCreated(node, app) {
        if (node.type === "ComfyUI-ImageCompare") {
           // console.log("ComfyUI-ImageCompare node created:", node);
        }
    }
});

console.log("%cComfyUI-ImageCompare: Registered", "color: cyan");
