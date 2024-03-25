import { spawn } from 'child_process';
import path from 'path';
import type { Result } from '../interfaces/Result.js';

export const runProcess = async (filePath: string): Promise<Result> => {
   const __dirname = path.resolve();
   let standardOutput = '';
   return new Promise(resolve => {
      const process = spawn(
         'node',
         [path.resolve(__dirname, filePath)],
         { shell: false },
      )
      process.stdout.on('data', data => {
         standardOutput += data;
         process.kill();
         resolve(JSON.parse(standardOutput));
      });
      process.stderr.on('data', data => {
         console.error(`Node ERROR: ${data}`);
         const output = data.toString().toLowerCase();
         let recoverableError;
         if (output.includes('timeout'))
            recoverableError = 'TIMEOUT';
         if (output.includes('reset'))
            recoverableError = 'RESET';
         if (recoverableError) {
            process.kill();
            resolve({
               errors: [recoverableError],
               function: '',
               messages: [],
               proceed: true,
            })
         }
      });
   })
}