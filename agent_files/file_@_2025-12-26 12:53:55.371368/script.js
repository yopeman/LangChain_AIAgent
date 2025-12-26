const input = document.getElementById('input');
const equals = document.getElementById('equals');
const clear = document.getElementById('clear');
const result = document.getElementById('result');

equals.addEventListener('click', () => {
    try {
        const calculation = eval(input.value);
        result.textContent = calculation;
    } catch (error) {
        result.textContent = 'Error';
    }
});

clear.addEventListener('click', () => {
    input.value = '';
    result.textContent = '';
});