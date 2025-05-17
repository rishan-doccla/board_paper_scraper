// Diagnostic script to debug the analysis modal

// Run when the document is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log("Diagnostic script loaded");

    // Hook into the modal being shown event
    const modalElement = document.getElementById('analysisModal');
    if (modalElement) {
        modalElement.addEventListener('show.bs.modal', function(event) {
            console.log("Modal is being shown");

            // Get the button that triggered the modal
            const button = event.relatedTarget;
            console.log("Button:", button);

            // Get the data attributes
            const analysisAttr = button.getAttribute('data-analysis');
            const titleAttr = button.getAttribute('data-title');

            console.log("Raw data-analysis attribute:", analysisAttr);
            console.log("Title attribute:", titleAttr);

            // Try to parse the JSON
            let analysisData;
            try {
                analysisData = JSON.parse(analysisAttr || '{}');
                console.log("Parsed analysis data:", analysisData);

                // Check if it has any keys
                const keys = Object.keys(analysisData);
                console.log("Terms found in analysis data:", keys);

                // Check each term's data structure
                keys.forEach(term => {
                    const termData = analysisData[term];
                    console.log(`Term "${term}" data:`, termData);

                    if (termData && termData.mentions) {
                        console.log(`Term "${term}" has ${termData.mentions.length} mentions`);
                        if (termData.mentions.length > 0) {
                            console.log(`First mention:`, termData.mentions[0]);
                        }
                    } else {
                        console.log(`Term "${term}" has no valid mentions array`);
                    }

                    if (termData && termData.summaries) {
                        console.log(`Term "${term}" has ${termData.summaries.length} summaries`);
                        if (termData.summaries.length > 0) {
                            console.log(`First summary:`, termData.summaries[0]);
                        }
                    } else {
                        console.log(`Term "${term}" has no valid summaries array`);
                    }
                });

            } catch (e) {
                console.error("Error parsing analysis data:", e);
                console.log("Raw data that caused the error:", analysisAttr);
            }
        });

        // Also hook the hidden event to see if there are any errors after it's closed
        modalElement.addEventListener('hidden.bs.modal', function(event) {
            console.log("Modal has been hidden");
        });
    } else {
        console.warn("Analysis modal element not found");
    }

    // Find all View Analysis buttons and check their data
    const analysisButtons = document.querySelectorAll('button[data-bs-target="#analysisModal"]');
    console.log(`Found ${analysisButtons.length} "View Analysis" buttons`);

    analysisButtons.forEach((button, index) => {
        console.log(`Button ${index + 1}:`);
        console.log(`  Title: ${button.getAttribute('data-title')}`);
        console.log(`  Analysis data length: ${button.getAttribute('data-analysis')?.length || 0} characters`);

        // Try to parse the analysis data
        try {
            const data = JSON.parse(button.getAttribute('data-analysis') || '{}');
            console.log(`  Analysis data parsed successfully with ${Object.keys(data).length} terms`);
        } catch (e) {
            console.error(`  Error parsing analysis data for button ${index + 1}:`, e);
        }
    });
});
