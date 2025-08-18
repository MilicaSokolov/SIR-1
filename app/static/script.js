document.getElementById('file-input').onchange = function(event) {
    if (event.target.files.length > 0) {
        var file = event.target.files[0];
        var reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('selected-image').src = e.target.result;
        };
        reader.readAsDataURL(file); 
    }
};

document.getElementById('upload-form').onsubmit = async function(event) {
    event.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('file-input');
    const loader = document.getElementById('loader');

    if (fileInput.files.length === 0) {
        alert('Please select a file.');
        return;
    }

    formData.append('file', fileInput.files[0]);

    loader.style.display = 'block'; 
    document.getElementById('pytorch-predicted-class').classList.add('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        loader.style.display = 'none';

        if (result.error) {
            console.log(result.error);
            alert('Error: ' + result.error);
        } else {
            const pytorchPredictedClassElement = document.getElementById('pytorch-predicted-class');
            pytorchPredictedClassElement.innerHTML = 'PyTorch Predicted class: <span style="color: #cc3d3d; font-weight: bold;">' + result.pytorch_class.toUpperCase() + '</span>';
            pytorchPredictedClassElement.classList.remove('hidden'); 
        }
    } catch (error) {
        loader.style.display = 'none';
        alert('An error occurred: ' + error.message);
    }
};
