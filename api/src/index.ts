import { dbClient } from './common/constants/dbClient.js';
import { createEndpoints } from './common/functions/createEndpoints.js';
import { initialize } from './common/functions/initialize.js';
import { retrieveWebBoxscores } from './common/functions/retrieveWebBoxscores.js';
//import { retrieveWebSchedules } from './common/functions/retrieveWebSchedules.js';

await dbClient.connect();
createEndpoints(initialize());
//retrieveWebSchedules();
retrieveWebBoxscores();