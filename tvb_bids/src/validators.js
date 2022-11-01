import validate from 'bids-validator';

/**
 * Function to create paragraphs of type warnings or errors and append them o a target
 * @param messages - list of strings
 * @param target - html element
 * @param type - string
 */
function buildParagraphs( messages, target, type ) {
    messages.forEach(err => {
        const errorParagraph = document.createElement('p');
        errorParagraph.style.color = type === 'errors'? 'red': 'yellow';
        errorParagraph.innerHTML = err.reason;
        target.appendChild(errorParagraph);
    });
}

/**
 * Creates a summary from a validation result and appends the result as a child to target html element
 * @param result - result object
 * @param target - target html element
 */
function buildResultSummary( result, target ) {
    if (result.errors.length > 0) {
        const description = '<p style="color: red;">Errors:</p>';
        target.innerHTML = description;
        buildParagraphs(result.errors, target, 'errors');
    }

    if (result.warnings.length > 0) {
        const description = document.createElement('p');
        description.style.color = 'yellow';
        description.innerHTML = 'Warnings:';
        target.appendChild(description);
        buildParagraphs(result.warnings, target, 'warnings');
    }
}

/**
 * Validates an object containing a list of uploaded files
 * @param selectedFiles
 */
function validateBidsDir( selectedFiles ) {
    const dirName = selectedFiles.list[0].webkitRelativePath.split('/')[0];
    const defaultConfig = `${dirName}/.bids-validator-config.json`;
    const resultTarget = document.querySelector('.errorMessage');
    resultTarget.innerHTML = '';
    validate.BIDS(
        selectedFiles.list,
        {
            verbose: true,
            ignoreWarnings: false,
            ignoreNiftiHeaders: false,
            ignoreSubjectConsistency: false,
            config: defaultConfig
        },
        (issues, summary) => {
            let result;
            if (issues === 'Invalid') {
                result = {
                    errors: 'Invalid',
                    summary,
                    status: 'validated',
                };
            } else {
                result = {
                    errors: issues.errors ? issues.errors : [],
                    warnings: issues.warnings ? issues.warnings : [],
                    summary,
                    status: 'validated',
                };
            }
            console.log('result: ', result);
            buildResultSummary(result, resultTarget);
            return result;
        },
    );
}

// add the event to directory input to validate the BIDS directory
const upload = document.getElementById('mri_data');
if (upload) {
    upload.addEventListener("change", (e) => {
        const selectedFiles = {list: e.target.files};
        validateBidsDir(selectedFiles);
    });
}