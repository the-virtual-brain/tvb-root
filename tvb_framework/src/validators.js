window.process = {
    env: {
        NODE_ENV: 'development'
    }
};

import validate from 'bids-validator';

const AdmZip = require("adm-zip");

const checkBidsValidity = ( bidsDatasetDir ) => {
    let valid = false;
    let errors = [];
    let warnings = [];
    let validationSummary = null;
    validate.BIDS( bidsDatasetDir, {verbose: true}, (issues, summary) => {
        validationSummary = summary
        if (issues === 'invalid') {
            errors = 'Invalid';
        } else {
            errors = issues.errors? issues.errors: [];
            warnings = issues.warnings ? issues.warnings: [];
            valid = true;
        }
    })
    return {
        valid: valid,
        errors: errors,
        warnings: warnings,
        summary: validationSummary
    };
}

/**
 *  Function to check if a zip file's contents respect the BIDS standard.
 *  First it reads the contents of a zip file then validates the structure
 * @param pathToBidsZip
 */
const validateBidsZip = ( pathToBidsZip ) => {
    console.log('bids zip: ', pathToBidsZip);
    const zip = AdmZip( pathToBidsZip, {} );
    // const zip = ZipFile( pathToBidsZip, checkBidsValidity );
    const summary = checkBidsValidity(zip.getEntries());
    console.log('summary: ', summary);
    alert('validity: ' + summary.valid);
};

const upload = document.getElementById('mri_data');
console.log('upload: ', upload);
if (upload) {
    upload.addEventListener("change", () => validateBidsZip(upload.value));
}
console.log('helloooooooou validate: ', validate);