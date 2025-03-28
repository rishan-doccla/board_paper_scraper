// NHS Board Paper Scraper JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const runCrawlerBtn = document.getElementById('runCrawlerBtn');
    const crawlerStatus = document.getElementById('crawlerStatus');
    const showNewOnlyCheckbox = document.getElementById('showNewOnly');
    const papersTableBody = document.getElementById('papersTableBody');
    const testUrlsBtn = document.getElementById('testUrlsBtn');
    const testUrlsStatus = document.getElementById('testUrlsStatus');
    const urlInput = document.getElementById('urlInput');
    
    // Run crawler button
    if (runCrawlerBtn) {
        runCrawlerBtn.addEventListener('click', function() {
            // Disable button and show status
            runCrawlerBtn.disabled = true;
            crawlerStatus.classList.remove('d-none');
            
            // Call the API to run the crawler
            fetch('/run-crawler', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                // Update the table with new results
                updatePapersTable(data.results.board_papers);
                
                // Show success message
                alert('Crawler completed successfully!');
                
                // Reload the page to show updated results
                window.location.reload();
            })
            .catch(error => {
                console.error('Error running crawler:', error);
                alert('Error running crawler. Check the console for details.');
            })
            .finally(() => {
                // Re-enable button and hide status
                runCrawlerBtn.disabled = false;
                crawlerStatus.classList.add('d-none');
            });
        });
    }
    
    // Test specific URLs button
    if (testUrlsBtn) {
        testUrlsBtn.addEventListener('click', function() {
            // Get the URLs from the input
            const urlsText = urlInput.value.trim();
            
            if (!urlsText) {
                alert('Please enter at least one URL to test.');
                return;
            }
            
            // Parse URLs with @ prefix or regular line-by-line format
            const urls = parseUrlInput(urlsText);
            
            if (urls.length === 0) {
                alert('Please enter at least one valid URL to test.');
                return;
            }
            
            // Disable button and show status
            testUrlsBtn.disabled = true;
            testUrlsStatus.classList.remove('d-none');
            
            // Call the API to test the URLs
            const payload = { urls: urls };
            console.log("Sending payload to server:", payload);
            
            fetch('/test-specific-urls', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // Update the table with new results
                    updatePapersTable(data.results.board_papers);
                    
                    // Show success message
                    alert(`Test completed successfully! Found ${data.results.board_papers.length} papers.`);
                    
                    // Reload the page to show updated results
                    window.location.reload();
                } else {
                    // Show error message
                    alert(`Error: ${data.message}`);
                }
            })
            .catch(error => {
                console.error('Error testing URLs:', error);
                alert('Error testing URLs. Check the console for details.');
            })
            .finally(() => {
                // Re-enable button and hide status
                testUrlsBtn.disabled = false;
                testUrlsStatus.classList.add('d-none');
            });
        });
    }
    
    // Function to parse URL input text that may contain @ prefixes
    function parseUrlInput(inputText) {
        console.log("Parsing URL input:", inputText);
        
        // First check if the input contains @ symbols (for the special format)
        if (inputText.includes('@')) {
            // Find all URLs with @ prefix
            const urlMatches = inputText.match(/@https?:\/\/[^\s]+/g);
            
            if (urlMatches) {
                console.log("Found @ prefixed URLs:", urlMatches);
                // Remove the @ prefix from each URL
                const parsedUrls = urlMatches.map(url => url.substring(1).trim());
                console.log("Parsed URLs:", parsedUrls);
                return parsedUrls;
            }
            console.log("No valid URLs found with @ prefix");
            return [];
        } else {
            // Regular line-by-line format (backward compatibility)
            const parsedUrls = inputText.split('\n')
                .map(url => url.trim())
                .filter(url => url.length > 0);
            console.log("Parsed line-by-line URLs:", parsedUrls);
            return parsedUrls;
        }
    }
    
    // Show new papers only checkbox
    if (showNewOnlyCheckbox) {
        showNewOnlyCheckbox.addEventListener('change', function() {
            const paperRows = document.querySelectorAll('.paper-row');
            
            if (this.checked) {
                // Show only new papers
                paperRows.forEach(row => {
                    if (!row.classList.contains('new-paper')) {
                        row.classList.add('hidden');
                    }
                });
            } else {
                // Show all papers
                paperRows.forEach(row => {
                    row.classList.remove('hidden');
                });
            }
        });
    }
    
    // Function to update the papers table
    function updatePapersTable(papers) {
        // Clear the table
        papersTableBody.innerHTML = '';
        
        if (papers && papers.length > 0) {
            // Sort papers chronologically by sort_date (yyyy-mm format)
            papers.sort((a, b) => {
                const dateA = a.sort_date || '9999-99';
                const dateB = b.sort_date || '9999-99';
                return dateA.localeCompare(dateB);
            });

            // Add papers to the table
            papers.forEach(paper => {
                const row = document.createElement('tr');
                row.className = `paper-row ${paper.is_new ? 'table-success new-paper' : ''}`;
                
                // Extract year and month for display
                let formattedDate = paper.date || '';
                
                // Try to improve the date display using sort_date if available
                if (paper.sort_date && paper.sort_date !== '9999-99') {
                    const parts = paper.sort_date.split('-');
                    if (parts.length === 2) {
                        const year = parts[0];
                        const monthNum = parseInt(parts[1]);
                        
                        if (monthNum > 0 && monthNum <= 12) {
                            const monthNames = [
                                'January', 'February', 'March', 'April', 'May', 'June',
                                'July', 'August', 'September', 'October', 'November', 'December'
                            ];
                            formattedDate = `${monthNames[monthNum-1]} ${year}`;
                        } else {
                            formattedDate = year;
                        }
                    }
                }
                
                row.innerHTML = `
                    <td>${paper.title}</td>
                    <td>${paper.organization}</td>
                    <td>${paper.org_type}</td>
                    <td>${formattedDate}</td>
                    <td>
                        <a href="${paper.url}" target="_blank" class="btn btn-sm btn-primary">View</a>
                    </td>
                `;
                
                papersTableBody.appendChild(row);
            });
        } else {
            // Show no papers message
            const row = document.createElement('tr');
            row.innerHTML = `
                <td colspan="5" class="text-center">No board papers found from 2024 onwards.</td>
            `;
            papersTableBody.appendChild(row);
        }
    }
}); 